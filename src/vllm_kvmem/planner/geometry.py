"""Model and cache geometry: what one token, one window and one request cost.

Exact numbers come from a server's ``cache-geometry.json`` (written at startup
into the capture directory). Without one, they are derived from the model's
``config.json``, the KV dtype and the number of speculative tokens, with the
same page layout rules vLLM's hybrid allocator uses (attention pages sized up to
the recurrent-state page).
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

DTYPE_BYTES = {'float32': 4, 'bfloat16': 2, 'float16': 2, 'fp8': 1, 'fp8_e4m3': 1,
               'float8_e4m3fn': 1}


@dataclass(frozen=True)
class Geometry:
    page_bytes: int                 # one KV-pool page (every cache group shares the size)
    pool_pages: int                 # usable pages in the GPU KV pool
    attention_page_tokens: int      # tokens of every attention layer held by one page
    attention_groups: int           # attention cache groups (each needs its own pages)
    recurrent_groups: int           # GDN cache groups
    recurrent_pages_per_request: int    # pages a running request holds for GDN states
    cpu_bytes_per_token: int        # CPU archive bytes per history token (all attention layers)
    recurrent_state_bytes: int      # one request's packed GDN state (CPU checkpoint payload)
    select_budget: int = 0          # active window tokens (0: unknown)
    gen_budget: int = 0             # tokens a window may grow past the select budget before compaction

    def window_tokens(self, history, budget=None):
        """Active window for ``history`` tokens: the whole history until it passes
        the select budget, then a sawtooth between budget and budget + gen_budget
        (each pressure refresh compacts back to the budget)."""
        budget = budget or self.select_budget
        if not budget or history <= budget:
            return history
        if not self.gen_budget:
            return budget
        return budget + (history - budget) % self.gen_budget

    def window_pages(self, tokens):
        """GPU pages for an active window of ``tokens`` (all attention groups)."""
        return self.attention_groups * math.ceil(tokens / self.attention_page_tokens)

    def running_pages(self, window_tokens):
        return self.window_pages(window_tokens) + self.recurrent_pages_per_request

    def window_bytes(self, tokens):
        """Bytes a swap-in moves for the window (CPU archive layout)."""
        return tokens * self.cpu_bytes_per_token

    def to_dict(self):
        return asdict(self)


def from_cache_geometry(path):
    """Exact geometry from a server capture's cache-geometry.json (schema 2)."""
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 2:
        raise ValueError('Unsupported cache-geometry schema')
    groups = data['groups']
    sizes = {g['page_bytes'] for g in groups}
    if len(sizes) != 1:
        raise ValueError('Cache groups with different page sizes are not modelled')
    attention = [g for g in groups if not g['recurrent']]
    recurrent = [g for g in groups if g['recurrent']]
    if not attention or len({g['block_tokens'] for g in attention}) != 1:
        raise ValueError('Attention groups must share one page size in tokens')
    spec = {g.get('speculative_blocks', 0) for g in recurrent}
    if len(spec) > 1:
        raise ValueError('Recurrent groups disagree on speculative state blocks')
    per_request = len(recurrent) * (1 + (spec.pop() if spec else 0))
    pool_pages = data['native_blocks'] - 1          # page 0 is vLLM's null block
    return Geometry(page_bytes=sizes.pop(), pool_pages=pool_pages,
                    attention_page_tokens=attention[0]['block_tokens'],
                    attention_groups=len(attention), recurrent_groups=len(recurrent),
                    recurrent_pages_per_request=per_request,
                    cpu_bytes_per_token=data['cpu_archive_bytes_per_token'],
                    recurrent_state_bytes=sum(g['payload_page_bytes'] for g in recurrent)
                        if recurrent else 0,
                    select_budget=data.get('select_budget', 0), gen_budget=data.get('gen_budget', 0))


def from_model_config(config_path, *, kv_pool_bytes, kv_dtype='fp8_e4m3', spec_tokens=0,
                      mtp_layers=None, select_budget=0, gen_budget=0):
    """Geometry derived from a Hugging Face config (hybrid GDN + attention models).

    Mirrors vLLM's hybrid page rule: every group uses one page size; attention
    pages hold as many tokens as fit the largest recurrent page, rounded up to a
    multiple of 16 tokens.
    """
    cfg = json.loads(Path(config_path).read_text())
    text = cfg.get('text_config', cfg)
    kinds = text.get('layer_types') or ['full_attention'] * text['num_hidden_layers']
    attn_layers = sum(k == 'full_attention' for k in kinds)
    gdn_layers = sum(k == 'linear_attention' for k in kinds)
    if mtp_layers is None:
        mtp_layers = text.get('mtp_num_hidden_layers', 0) if spec_tokens else 0
    kv_elem = DTYPE_BYTES[kv_dtype]
    head_dim = text.get('head_dim') or text['hidden_size'] // text['num_attention_heads']
    attn_layer_token = 2 * text['num_key_value_heads'] * head_dim * kv_elem
    cpu_bytes_per_token = (attn_layers + mtp_layers) * attn_layer_token
    if gdn_layers:
        ssm_elem = DTYPE_BYTES[text.get('mamba_ssm_dtype', 'float32')]
        conv_elem = DTYPE_BYTES[text.get('dtype', text.get('torch_dtype', 'bfloat16'))]
        k_dim = text['linear_num_key_heads'] * text['linear_key_head_dim']
        v_dim = text['linear_num_value_heads'] * text['linear_value_head_dim']
        conv_dim = 2 * k_dim + v_dim
        ssm = text['linear_num_value_heads'] * text['linear_key_head_dim'] * text['linear_value_head_dim'] * ssm_elem
        conv = conv_dim * (text['linear_conv_kernel_dim'] - 1 + spec_tokens) * conv_elem
        gdn_layer_state = ssm + conv
    else:
        gdn_layer_state = 0
    # vLLM groups layers so every group has as many layers as the attention group.
    group_layers = max(1, attn_layers + mtp_layers)
    recurrent_groups = math.ceil(gdn_layers / group_layers) if gdn_layers else 0
    recurrent_page = group_layers * gdn_layer_state
    page_tokens = 16 * math.ceil(recurrent_page / (group_layers * attn_layer_token) / 16) if gdn_layers else 16
    page_bytes = group_layers * attn_layer_token * page_tokens
    return Geometry(page_bytes=page_bytes, pool_pages=kv_pool_bytes // page_bytes - 1,
                    attention_page_tokens=page_tokens, attention_groups=1,
                    recurrent_groups=recurrent_groups,
                    recurrent_pages_per_request=recurrent_groups * (1 + spec_tokens),
                    cpu_bytes_per_token=cpu_bytes_per_token,
                    recurrent_state_bytes=gdn_layers * gdn_layer_state,
                    select_budget=select_budget, gen_budget=gen_budget)

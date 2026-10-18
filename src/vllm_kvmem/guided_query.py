"""Private, greedy Query rewrite inside a fenced native refresh transaction.

The scheduler reserves scratch pages within K+G. No private token is sampled by
the public sampler or submitted to the historical authority. The caller restores
all native recurrent state before assembling/replaying the public Query.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch

from .capture import capture_to


PRIVATE_PROMPT = (
    '</think>\n<|im_end|>\n<|im_start|>user\n'
    'PRIVATE MEMORY RETRIEVAL TASK\n\n'
    'Based on the original coding task, current progress, and recent tool results visible '
    'above, output one compact standalone request describing exactly what earlier evidence is '
    'needed to continue correctly. Preserve concrete file names, symbols, errors, tests, constraints, '
    'and unfinished steps. Do not solve the task, call a tool, emit JSON/XML, explain, or reason '
    'out loud. Output exactly one grammatical line.'
    '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nRetrieval query: '
)


class GuidedQueryPolicy:
    """qw3 guided Query acceptance, shared by synchronous/cooperative probes.

    Only naturally terminated, non-structured content replaces public Query Q.
    Prefix separators are consumed without contributing Q. Unlike qw3's private
    allocator, vLLM reserves a fixed token span: ignored prefix tokens count
    toward ``max_tokens`` too, so the policy cannot consume extra K/G scratch.
    """
    PREFIX_GUARD = 16
    LEADING_WRAPPERS = frozenset(('"', "'", '`', '“', '‘'))

    def __init__(self, tokenizer, max_tokens, *, direct=False):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('Guided Query requires a positive consumed-token limit')
        self.tokenizer, self.max_tokens, self.direct = tokenizer, max_tokens, direct
        eos = tokenizer.eos_token_id
        self.stop_tokens = {eos} if isinstance(eos,int) else set(eos or ())
        unknown = getattr(tokenizer,'unk_token_id',None)
        for marker in ('<|im_end|>', '</think>'):
            token = tokenizer.convert_tokens_to_ids(marker)
            if isinstance(token,int) and token >= 0 and token != unknown:
                self.stop_tokens.add(token)
        self.consumed = []
        self.captured = []
        self.ignored_prefix = 0
        self.pending = None
        self.finished = self.natural_termination = False
        self.fallback = None

    def prepare(self, token):
        """Return capture flag, or None when no further forward is allowed."""
        if self.finished or self.pending is not None:
            raise RuntimeError('Guided Query transition already prepared or finished')
        # Match qw3's loop: exhausting the cap is not natural completion. The
        # terminal token must be observed while the private decode still has
        # headroom; neither this branch nor a stop consumes another KV slot.
        if type(token) is not int or token < 0 or len(self.consumed) >= self.max_tokens:
            self._finish()
            return None
        if token in self.stop_tokens:
            self.natural_termination = True
            self._finish()
            return None
        piece = self.tokenizer.decode([token],skip_special_tokens=False)
        wrapper = self.direct and piece.strip() in self.LEADING_WRAPPERS
        content = bool(piece) and not piece.isspace() and not wrapper
        capture = bool(self.captured) or content
        if not capture and self.ignored_prefix >= self.PREFIX_GUARD:
            self._finish()
            return None
        self.pending = token,capture
        return capture

    def commit(self):
        """Publish a private row only after its model forward succeeded."""
        if self.pending is None or self.finished:
            raise RuntimeError('Guided Query has no pending consumed row')
        token,capture = self.pending
        self.consumed.append(token)
        if capture:
            self.captured.append(token)
        else:
            self.ignored_prefix += 1
        self.pending = None

    def _finish(self):
        text = self.tokenizer.decode(self.captured,skip_special_tokens=False).strip()
        lower = text.lower()
        structured = (any(marker in lower for marker in ('<answer','tool_call','<tool','```')) or
                      bool(text) and text[0] in '{[')
        self.fallback = ('structured-output' if structured else 'empty-output' if not text else
                         'unterminated-output' if not self.natural_termination else None)
        self.finished = True

    def report(self):
        if not self.finished:
            raise RuntimeError('Guided Query diagnostics require a completed decision')
        return dict(guided_query_tokens=len(self.consumed),  # Existing consumed-token metric.
            guided_consumed_tokens=len(self.consumed),guided_captured_tokens=len(self.captured),
            guided_published_tokens=0 if self.fallback else len(self.captured),
            guided_ignored_prefix_tokens=self.ignored_prefix,
            guided_natural_termination=self.natural_termination,guided_fallback=self.fallback)


@dataclass(frozen=True)
class GuidedQueryResult:
    queries: object
    prompt_tokens: int
    diagnostics: dict


def collect_guided_query(tokenizer, max_tokens, proposed, begin, forward, *, direct=False):
    """Consume bounded greedy transitions and stage Q until policy acceptance."""
    policy = GuidedQueryPolicy(tokenizer,max_tokens,direct=direct)
    rows = []
    while True:
        capture = policy.prepare(proposed)
        if capture is None:
            break
        proposed,q = forward([proposed],begin+len(policy.consumed))
        policy.commit()
        if capture:
            rows.append(q)
    queries = torch.cat(rows,dim=1) if rows and policy.fallback is None else None
    return queries,policy.report()


def private_batch(source, request_index, tokens, begin, device):
    """Independent buffers; never overwrite native public token/sampler inputs."""
    n = len(tokens)
    idx = source.idx_mapping[request_index:request_index+1].clone()
    starts = np.array([0, n], dtype=np.int32)
    return replace(source, req_ids=[source.req_ids[request_index]], num_reqs=1,
        num_reqs_after_padding=1, idx_mapping=idx,
        idx_mapping_np=source.idx_mapping_np[request_index:request_index+1].copy(),
        expanded_idx_mapping=idx, expanded_local_pos=torch.zeros(1,device=device,dtype=torch.int32),
        num_scheduled_tokens=np.array([n],dtype=np.int32), num_tokens=n,
        num_tokens_after_padding=n, num_draft_tokens=0, num_draft_tokens_per_req=None,
        query_start_loc=torch.tensor(starts,device=device), query_start_loc_np=starts,
        seq_lens=torch.tensor([begin+n],device=device,dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([begin+n],dtype=torch.int32),
        dcp_local_seq_lens=None, num_computed_tokens_np=np.array([begin],dtype=np.int32),
        prefill_len_np=np.array([begin+n],dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([begin],dtype=np.int32),
        is_prefilling_np=np.ones(1,dtype=np.bool_), has_prefill=True,
        input_ids=torch.tensor(tokens,device=device,dtype=torch.int32),
        positions=torch.arange(begin,begin+n,device=device,dtype=torch.int64),
        is_padding=torch.zeros(n,device=device,dtype=torch.bool),
        logits_indices=torch.tensor([n-1],device=device,dtype=torch.int64),
        cu_num_logits=torch.tensor([0,1],device=device,dtype=torch.int32),
        cu_num_logits_np=np.array([0,1],dtype=np.int32),
        has_structured_output_reqs=False,prompt_lens=None,max_query_len=n,fast_prefill=None,
        dcp_local_seq_lens_cpu_upper_bound=None)


def guided_query(runner, original, source, index, begin, reserve, max_tokens):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.tokenizers import get_tokenizer
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    tokenizer = getattr(runner, '_kvmem_guided_tokenizer', None)
    if tokenizer is None:
        cfg = runner.vllm_config.model_config
        tokenizer = get_tokenizer(cfg.tokenizer, tokenizer_mode=cfg.tokenizer_mode,
                                  trust_remote_code=cfg.trust_remote_code)
        runner._kvmem_guided_tokenizer = tokenizer
    prompt = tokenizer.encode(PRIVATE_PROMPT, add_special_tokens=False)
    if not prompt or len(prompt) + max_tokens > reserve:
        raise ValueError('Private Query prompt exceeds its scheduler scratch reservation')
    saved_batch, saved_window = runner._kvmem_last_batch, runner._kvmem_window_batch

    def forward(tokens, position):
        batch = private_batch(source,index,tokens,position,runner.device)
        tables, slots = runner.prepare_attn(batch)
        metadata = runner.model_state.prepare_attn(batch,CUDAGraphMode.NONE,tables,slots,
                                                   runner.attn_groups,runner.kv_cache_config)
        model_inputs = {'input_ids':batch.input_ids,'positions':batch.positions,
                        'inputs_embeds':None,'intermediate_tensors':None,
                        **runner.model_state.prepare_inputs(batch,runner.req_states)}
        captures = []
        with set_forward_context(metadata,runner.vllm_config,num_tokens=len(tokens),
                slot_mapping=build_slot_mappings_by_layer(slots,runner.kv_cache_config),
                skip_compiled=True,is_padding=batch.is_padding), capture_to(captures.append):
            result = original(**model_inputs)
        if getattr(runner,'_kvmem_graph',False):
            captures = runner._kvmem_graph_captures
        hidden = result[0] if isinstance(result,tuple) else result
        logits = runner.model.compute_logits(hidden[-1:])
        next_token = int(logits.argmax(dim=-1).item())
        if tuple(c.name for c in captures) != tuple(runner._kvmem_layers):
            raise RuntimeError('Guided Query missed a target attention layer')
        q = torch.stack([c.query[:len(tokens)] for c in captures]).clone()
        return next_token, q

    try:
        token, _ = forward(prompt, begin)
        queries,diagnostics = collect_guided_query(tokenizer,max_tokens,token,begin+len(prompt),forward)
        return GuidedQueryResult(queries,len(prompt),diagnostics)
    finally:
        runner._kvmem_last_batch, runner._kvmem_window_batch = saved_batch, saved_window

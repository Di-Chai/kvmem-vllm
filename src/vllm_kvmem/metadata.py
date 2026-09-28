"""Per-group attention lengths while recurrent history accounting stays intact."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowLengths:
    seq_lens: torch.Tensor
    cpu_upper_bound: torch.Tensor
    max_seq_len: int

    def __post_init__(self):
        if (self.seq_lens.ndim != 1 or self.seq_lens.dtype != torch.int32 or
                self.cpu_upper_bound.shape != self.seq_lens.shape or
                self.cpu_upper_bound.device.type != "cpu" or
                self.cpu_upper_bound.dtype != torch.int32 or self.max_seq_len < 0):
            raise ValueError("Invalid attention-window length buffers")
        if self.cpu_upper_bound.numel() and (
                self.cpu_upper_bound.min().item() < 0 or
                self.cpu_upper_bound.max().item() > self.max_seq_len):
            raise ValueError("Window upper bounds exceed configured attention extent")


class WorkingSetMetadata:
    """Wrap native metadata, preserving all GDN/speculation-specific fields."""
    def __init__(self, native, attention_groups: dict[int, WindowLengths]):
        self.native = native
        self.attention_groups = attention_groups

    def get_extra_common_attn_kwargs(self, kv_cache_group_id: int, num_reqs: int):
        kwargs = (dict(self.native.get_extra_common_attn_kwargs(kv_cache_group_id, num_reqs))
                  if self.native is not None else {})
        window = self.attention_groups.get(kv_cache_group_id)
        if window is not None:
            if num_reqs > window.seq_lens.numel():
                raise ValueError("Window lengths missing requests")
            kwargs.update(seq_lens=window.seq_lens[:num_reqs],
                          seq_lens_cpu_upper_bound=window.cpu_upper_bound[:num_reqs],
                          max_seq_len=window.max_seq_len)
        return kwargs

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs: int):
        return (self.native.get_extra_attn_kwargs(attn_metadata_builder, num_reqs)
                if self.native is not None else {})

"""Small proposed upstream hook: per-group attention lengths in V2 metadata.

The existing model-specific metadata interface supplies the overrides. No model
or scheduler computations change for callers which do not supply these fields.
Not activated by plugin discovery; engine bridge integration is still pending.
"""
import hashlib

SOURCE_SHA256 = "b9b81e59dda2720b1f9c434471381f879228832443e52756e469b4ad4905888f"


def patch_metadata_source(source: bytes) -> str:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ImportError("Unqualified vLLM V2 attention metadata source")
    text = source.decode()
    old = """            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,"""
    new = """            seq_lens=common_attn_metadata_extra_kwargs.pop("seq_lens", seq_lens),
            seq_lens_cpu_upper_bound=common_attn_metadata_extra_kwargs.pop(
                "seq_lens_cpu_upper_bound", seq_lens_cpu_upper_bound
            ),
            max_seq_len=common_attn_metadata_extra_kwargs.pop("max_seq_len", max_seq_len),"""
    if text.count(old) != 1:
        raise ImportError("Ambiguous per-group metadata patch")
    return text.replace(old, new)

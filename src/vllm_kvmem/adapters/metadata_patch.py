"""Small proposed upstream hook: per-group attention lengths in V2 metadata.

The existing model-specific metadata interface supplies the overrides. No model
or scheduler computations change for callers which do not supply these fields.
Enabled only by the explicit KVMEM_WINDOW_METADATA compatibility option.
"""
def patch_metadata_source(source: bytes) -> str:
    text = source.decode()
    # Vendor runtimes may build/reuse metadata inside a deeper group loop.
    # Check the actual constructor and its unmodified arguments, not indentation.
    import ast
    calls = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'CommonAttentionMetadata']
    if len(calls) != 1:
        raise ImportError('Ambiguous per-group metadata patch')
    keywords = {item.arg: item.value for item in calls[0].keywords}
    for name in ('seq_lens', 'seq_lens_cpu_upper_bound', 'max_seq_len', 'positions'):
        value = keywords.get(name)
        if not isinstance(value, ast.Name) or value.id != name:
            raise ImportError('Ambiguous per-group metadata patch')
    for name in ('seq_lens', 'seq_lens_cpu_upper_bound', 'max_seq_len'):
        old = f'{name}={name},'
        if text.count(old) != 1:
            raise ImportError('Ambiguous per-group metadata patch')
        text = text.replace(old, f'{name}=common_attn_metadata_extra_kwargs.pop("{name}", {name}),')
    old_positions = "positions=positions,"
    if text.count(old_positions) != 1:
        raise ImportError("Ambiguous per-group position patch")
    text = text.replace(old_positions,
                        'positions=common_attn_metadata_extra_kwargs.pop("positions", positions),')
    anchor = "    attn_metadata: dict[str, Any] = {}"
    if text.count(anchor) != 1:
        raise ImportError("Ambiguous attention-window context hook")
    return text.replace(anchor, """    from vllm_kvmem.serving.window_batch import contextual_metadata
    model_specific_attn_metadata = contextual_metadata(model_specific_attn_metadata)
""" + anchor)

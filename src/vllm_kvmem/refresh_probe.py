"""Actual native-page roundtrip diagnostic, not a replacement serving backend.

Exercise the full immutable-source pipeline without changing the selected set:
capture -> bounded repository -> async archive -> plan -> fused native-page
assembly -> completion/commit. Synchronization is explicit for this diagnostic.
"""
import math

import torch

from .assembly import assemble_fp8
from .codec import KVFormat, native_vllm_fp8_views, quantize
from .config import KVMemConfig
from .memory import RequestMemory
from .storage import AsyncArchive


def rebuild_native_batch(runner, captured, output):
    batch = runner._kvmem_last_batch
    if batch.num_reqs != 1 or batch.num_draft_tokens:
        raise ValueError("Initial archive/rebuild probe requires one non-speculative request")
    tokens = batch.num_tokens
    if batch.num_computed_tokens_np[0] != 0:
        raise ValueError("Archive/rebuild probe must start at the initial prefill")
    if len({(c.key.shape[1], c.key.shape[2]) for c in captured}) != 1:
        raise ValueError("Probe payload requires uniform target KV head geometry")
    config = KVMemConfig()
    heads, dim = captured[0].key.shape[1:]
    block_tokens = config.block_tokens
    layers = len(captured)
    raw_bytes = layers * block_tokens * heads * dim * 2
    value_bytes = raw_bytes // 2
    record_bytes = raw_bytes + value_bytes
    memory = RequestMemory(batch.req_ids[0], config)
    spans = memory.begin_append(tokens, 0)
    if spans is None:
        raise RuntimeError("Probe repository admission failed")
    module_by_name = dict(runner.model.named_modules())
    layer_names = {name: group_id for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                   for name in group.layer_names}
    raw = torch.stack([c.key[:tokens].cpu() for c in captured])
    quantized_values = []
    native_views = []
    slots_by_layer = []
    positions_by_layer = []
    before = []
    for c in captured:
        module = module_by_name[c.name]
        attn = module.attn
        if attn.kv_cache_dtype != "fp8_e4m3":
            raise ValueError("Native rebuild probe requires FP8 e4m3")
        k, v = native_vllm_fp8_views(attn.kv_cache, heads, dim)
        slots = runner.block_tables.slot_mappings[layer_names[attn.layer_name], :tokens].clone()
        positions = c.positions[..., :tokens]
        if positions.ndim == 2:
            if not torch.equal(positions, positions[:1].expand_as(positions)):
                raise ValueError("Multimodal MRoPE remapping is not yet qualified")
            positions = positions[0]
        native_views.append((k, v, module))
        slots_by_layer.append(slots)
        positions_by_layer.append(positions.clone())
        page, offset = slots // k.shape[1], slots % k.shape[1]
        before.append((k.view(torch.uint8)[page, offset].clone(), v.view(torch.uint8)[page, offset].clone()))
        fmt = KVFormat(torch.float8_e4m3fn, dim, heads, k.shape[1])
        quantized_values.append(quantize(c.value[:tokens], fmt, attn._v_scale).view(torch.uint8).cpu())
    values = torch.stack(quantized_values)
    prepared = []
    for span in spans:
        start, count = span.history_begin, span.token_count
        padded_k = torch.zeros(layers, block_tokens, heads, dim, dtype=raw.dtype)
        padded_v = torch.zeros(layers, block_tokens, heads, dim, dtype=torch.uint8)
        padded_k[:, :count].copy_(raw[:, start:start+count])
        padded_v[:, :count].copy_(values[:, start:start+count])
        prepared.append(padded_k.view(torch.uint8).numpy().tobytes() + padded_v.numpy().tobytes())
    # Prepared CPU authority now owns all valid token payloads.
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    memory.attach_completion(event)
    assert memory.commit_append(tokens)
    leases = [memory.repository.pin(span.handle) for span in spans]
    archive = AsyncArchive(str(output / "archive"), record_bytes, max(2, math.ceil(tokens / block_tokens)))
    write_stats = archive.write(memory.request_id, leases, b"".join(prepared)).result(timeout=30)
    restored, read_stats = archive.read(memory.request_id, leases).result(timeout=30)
    plan = memory.select({span.handle.slot: 1.0 for span in spans})
    if plan.total_window_tokens != tokens:
        raise ValueError("Identity qualification must retain every input token")
    restored_k, restored_v = [], []
    for index, span in enumerate(spans):
        start = index * record_bytes
        rk = torch.frombuffer(restored, dtype=raw.dtype, count=raw_bytes//2, offset=start)
        rv = torch.frombuffer(restored, dtype=torch.uint8, count=value_bytes, offset=start + raw_bytes)
        restored_k.append(rk.view(layers, block_tokens, heads, dim)[:, :span.token_count])
        restored_v.append(rv.view(layers, block_tokens, heads, dim)[:, :span.token_count])
    restored_k = torch.cat(restored_k, dim=1).to(runner.device)
    restored_v = torch.cat(restored_v, dim=1).to(runner.device).view(torch.float8_e4m3fn)
    reports = []
    for i, (k, v, module) in enumerate(native_views):
        slots = slots_by_layer[i]
        assemble_fp8(restored_k[i], restored_v[i], positions_by_layer[i], slots,
                     module.rotary_emb.cos_sin_cache, k, v, module.attn._k_scale)
        page, offset = slots // k.shape[1], slots % k.shape[1]
        k_equal = torch.equal(k.view(torch.uint8)[page, offset], before[i][0])
        v_equal = torch.equal(v.view(torch.uint8)[page, offset], before[i][1])
        reports.append({"layer": captured[i].name, "key_byte_equal": k_equal,
                        "value_byte_equal": v_equal, "page_tokens": k.shape[1],
                        "native_shape": list(module.attn.kv_cache.shape),
                        "native_strides": list(module.attn.kv_cache.stride())})
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    memory.attach_completion(done)
    assert memory.commit_selection()
    for lease in leases:
        lease.reset()
    assert memory.close()
    for slot in range(config.repository_blocks):
        handle = memory.repository.retired_at(slot)
        if handle.generation:
            memory.repository.reclaim(handle, lambda record: archive.reclaim(memory.request_id, record))
    assert archive.close()
    return {"tokens": tokens, "blocks": len(spans), "layers": reports,
            "write": write_stats, "read": read_stats,
            "all_byte_equal": all(r["key_byte_equal"] and r["value_byte_equal"] for r in reports)}

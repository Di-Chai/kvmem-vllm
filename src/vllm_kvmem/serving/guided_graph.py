"""One bounded, transaction-local CUDA graph for private greedy Query decode.

Native attention builders prepare the initial page tables and kernel layout.
The graph updates working positions/lengths/slots, runs the model and publishes
greedy argmax and sampled Q. No owner metadata survives the private branch.
"""
from dataclasses import fields, is_dataclass
from enum import Enum
import time

import torch


class UnsupportedGuidedGraph(ValueError):
    pass


def metadata_signature(value):
    """Reject changed addresses/scalars before replaying a captured attention."""
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cuda':
            raise UnsupportedGuidedGraph('cpu-attention-tensor')
        return ('tensor', value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                value.dtype, value.device)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, metadata_signature(getattr(value, f.name)))
                                  for f in fields(value)))
    if isinstance(value, dict):
        return tuple((key, metadata_signature(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(metadata_signature(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, Enum, torch.dtype)):
        return value
    raise UnsupportedGuidedGraph('attention-object-' + type(value).__name__)


def restore_tensors(destinations, snapshot):
    # Validate the entire schema before modifying any live recurrent state.
    if destinations.keys() != snapshot.keys() or any(
            (v.shape, v.dtype, v.device) != (snapshot[k].shape, snapshot[k].dtype, snapshot[k].device)
            for k, v in destinations.items()):
        raise ValueError('Private graph warmup changed the recurrent schema')
    for name, value in destinations.items():
        value.copy_(snapshot[name])


class PrivateDecodeGraph:
    MAX_BYTES = 256 * 1024 * 1024
    MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024

    def __init__(self, *, measure_gpu=False):
        self.graph = self.signature = self.outputs = None
        self.replays = self.allocated_bytes = 0
        self.setup_seconds = 0.0
        self.snapshot_bytes = 0
        self.fallback = None
        self.measure_gpu = measure_gpu
        self.gpu_events = None
        self.gpu_samples = self.gpu_ms = 0

    def capture(self, run, metadata, model_inputs, slot_mapping, recurrent):
        self.signature = metadata_signature((metadata, model_inputs, slot_mapping))
        self.snapshot_bytes = sum(v.numel()*v.element_size() for v in recurrent.values())
        if self.snapshot_bytes > self.MAX_SNAPSHOT_BYTES:
            self.fallback = 'warmup-state-memory-limit'
            return
        started = time.perf_counter()
        snapshot = {name: value.clone() for name, value in recurrent.items()}
        stream = torch.cuda.current_stream()
        warmup = None
        # Restore every supplied touched row/counter before the first real
        # transition. The caller qualifies the native non-speculative layout.
        try:
            warmup = torch.cuda.Stream()
            warmup.wait_stream(stream)
            with torch.cuda.stream(warmup):
                run()
            stream.wait_stream(warmup)
            restore_tensors(recurrent, snapshot)
            stream.synchronize()
            allocated = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = run()
            self.allocated_bytes = max(0, torch.cuda.memory_allocated() - allocated)
            if self.allocated_bytes > self.MAX_BYTES:
                graph.reset()
                self.fallback = 'graph-memory-limit'
            else:
                self.graph, self.outputs = graph, outputs
        finally:
            if warmup is not None:
                # Also fence a warmup that raised after enqueueing GPU work.
                stream.wait_stream(warmup)
            restore_tensors(recurrent, snapshot)
            stream.synchronize()
            self.setup_seconds = time.perf_counter() - started

    def replay(self, metadata, model_inputs, slot_mapping, *, prepared=False):
        if not prepared and metadata_signature((metadata, model_inputs, slot_mapping)) != self.signature:
            self.fallback = 'attention-address-or-layout-changed'
            return None
        sample = self.measure_gpu and self.gpu_samples < 8
        if sample:
            if self.gpu_events is None:
                self.gpu_events = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            self.gpu_events[0].record()
        self.graph.replay()
        if sample:
            self.gpu_events[1].record()
        self.replays += 1
        token, queries = self.outputs
        # The graph writes the same output buffer on every replay.
        proposed = int(token.item())
        if sample:
            self.gpu_ms += self.gpu_events[0].elapsed_time(self.gpu_events[1])
            self.gpu_samples += 1
        return proposed, queries.clone()

    def close(self):
        if self.graph is not None:
            torch.cuda.current_stream().synchronize()
            self.graph.reset()
            self.graph = self.outputs = None

    def report(self):
        return dict(guided_execution_mode='cuda_graph' if self.replays else 'eager',
                    guided_graph_replays=self.replays, guided_graph_bytes=self.allocated_bytes,
                    guided_graph_warmup_state_bytes=self.snapshot_bytes,
                    guided_graph_setup_seconds=self.setup_seconds,
                    guided_graph_sampled_gpu_ms=self.gpu_ms/self.gpu_samples if self.gpu_samples else None,
                    guided_execution_fallback=self.fallback)

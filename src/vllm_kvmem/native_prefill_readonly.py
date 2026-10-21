"""Component-only immutable metadata checks for static FA2's __ldg experiment.

The service never requests this module. The component caller owns metadata and
uses one CUDA stream for arm/upload, FA2, validation, and later mutation. Address
checks prove absence of writes through this call's tensor arguments; they cannot
make an external concurrent write safe. The existing bank's single-stream frame
lease remains required. No pointer is marked restrict and no data is read back.
"""
from collections.abc import Sequence


class ReadonlyMetadataIneligible(ValueError):
    pass


def _extent(tensor, *, allow_empty=True):
    if tensor is None:
        return None
    try:
        device = tensor.device
        if device.type != 'cuda':
            raise ReadonlyMetadataIneligible('metadata guard requires CUDA tensors')
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
        size = tensor.element_size()
        pointer = tensor.data_ptr()
        if (len(shape) != len(strides) or type(size) is not int or size <= 0 or
                any(type(n) is not int or n < 0 for n in shape) or
                any(type(n) is not int or n < 0 for n in strides)):
            raise ReadonlyMetadataIneligible('unsupported tensor extent/stride')
        if 0 in shape:
            if not allow_empty:
                raise ReadonlyMetadataIneligible('empty immutable metadata')
            return str(device), pointer, pointer
        if type(pointer) is not int or pointer <= 0:
            raise ReadonlyMetadataIneligible('invalid tensor pointer')
        # Positive-stride byte enclosure is conservative even for a sliced or
        # interleaved writable tensor; reject if disjointness cannot be proved.
        length = (1 + sum((n - 1) * stride for n, stride in zip(shape, strides))) * size
        return str(device), pointer, pointer + length
    except (AttributeError, TypeError) as error:
        raise ReadonlyMetadataIneligible('unknown tensor metadata') from error


def _overlaps(a, b):
    return a is not None and b is not None and a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def qualify_readonly_arguments(args):
    """Validate the original 24-argument static ABI without touching contents.

    Both workspaces are conservatively treated as writable, even though this
    execution mainly reads the integer plan workspace. Split-K temporaries lie
    in the float workspace. Native O/LSE and KVMem's LSE are the remaining writes.
    Read/read alias (e.g. q_indptr and a region of rows) is harmless and allowed.
    """
    if len(args) != 24:
        raise ReadonlyMetadataIneligible('unknown static FA2 ABI')
    plan = args[2]
    if (not isinstance(plan, Sequence) or len(plan) != 15 or
            type(plan[13]) not in (int, bool) or plan[13]):
        raise ReadonlyMetadataIneligible('graph or unknown static FA2 plan')
    if type(args[22]) is not int or args[22] != 0:
        raise ReadonlyMetadataIneligible('mixed decode prefix is unsupported')
    if type(args[14]) is not int or args[14] != -1:
        raise ReadonlyMetadataIneligible('sliding-window experiment is unsupported')
    reads = []
    for name, index, allowed_dtype in (
        ('q_indptr', 6, ('torch.int32', 'torch.int64')),
        ('stat_rows', 16, ('torch.int32',)),
    ):
        tensor = args[index]
        extent = _extent(tensor, allow_empty=False)
        if (len(tensor.shape) != 1 or tuple(tensor.stride()) != (1,) or
                str(tensor.dtype) not in allowed_dtype):
            raise ReadonlyMetadataIneligible(name + ' must be a contiguous integer vector')
        reads.append((name, extent))
    if reads[0][1][0] != reads[1][1][0]:
        raise ReadonlyMetadataIneligible('metadata is on different CUDA devices')
    writes = [(name, _extent(args[index])) for name, index in (
        ('float_workspace', 0), ('int_workspace', 1), ('attention_output', 10),
        ('native_row_lse', 11), ('statistics_lse', 19))]
    for name, extent in writes:
        if extent is not None and extent[0] != reads[0][1][0]:
            raise ReadonlyMetadataIneligible(name + ' is on a different CUDA device')
        for reader, read in reads:
            if _overlaps(read, extent):
                raise ReadonlyMetadataIneligible(reader + ' overlaps writable ' + name)
    return dict(readonly_regions=tuple(reads), writable_regions=tuple(writes))


class CheckedReadonlyPrefillModule:
    """Keep the experiment explicit, with per-call geometry/alias checks.

    This wrapper changes no stream and adds no fence/allocation. Static frame
    qualification still belongs to NativeAttentionStats. Failed qualification
    raises before any device work rather than silently executing an unsafe load.
    """
    def __init__(self, module, *, library_path, overlay):
        self.module = module
        self.library_path, self.overlay = str(library_path), str(overlay)
        self.checked_dispatches = 0

    def paged_run(self, *args):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise ReadonlyMetadataIneligible('readonly component never captures CUDA graphs')
        qualify_readonly_arguments(args)
        result = self.module.paged_run(*args)
        self.checked_dispatches += 1
        return result

    def __getattr__(self, name):
        if name == 'ragged_run':
            raise ReadonlyMetadataIneligible('readonly component supports paged FA2 only')
        return getattr(self.module, name)

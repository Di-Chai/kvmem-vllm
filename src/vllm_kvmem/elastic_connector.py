"""Native full-context tiering, preserving upstream graphs and MTP handling.

No KVMem selected-window/capture hooks are enabled by this connector. Native
aligned GDN checkpoints and speculative tails retain their upstream contracts.
"""
import json
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from .integration import check_engine_version


class KVMemElasticConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        check_engine_version()
        if (vllm_config.cache_config.mamba_cache_mode != 'align'
                or not vllm_config.cache_config.enable_prefix_caching):
            raise ValueError('Elastic full-context tiering requires native aligned prefix checkpoints')
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config
            directory = Path(extra['kvmem_report_dir'])
            directory.mkdir(parents=True, exist_ok=True)
            # The V2 runner overlays descriptors on one shared allocation.
            # A group's page consumes a whole pool slot, including padding;
            # its tensor payload alone understates the admission reservation.
            sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
            if len(sizes) != 1 or kv_cache_config.num_blocks <= 0:
                raise ValueError('Expected one native V2 KV backing allocation')
            capacity = sizes.pop()
            pool_page_bytes, remainder = divmod(capacity, kv_cache_config.num_blocks)
            if remainder:
                raise ValueError('Native KV backing is not an integral shared page pool')
            groups = []
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                page_bytes = spec.page_size_bytes
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    representative = spec.first_spec
                else:
                    representative = spec
                    page_bytes *= len(group.layer_names)
                groups.append({'block_tokens': representative.block_size,
                    'page_bytes': pool_page_bytes, 'payload_page_bytes': page_bytes,
                    'layers': list(group.layer_names),
                    'recurrent': isinstance(representative, MambaSpec),
                    'speculative_blocks': getattr(representative, 'num_speculative_blocks', 0)})
            manifest = {'schema': 1, 'groups': groups, 'capacity_bytes': capacity,
                'native_blocks': kv_cache_config.num_blocks,
                'cpu_offload_bytes': extra['cpu_bytes_to_use'],
                'semantic_mode': 'full_context_native_aligned_checkpoint',
                'partial_tail': 'recompute after latest matching aligned checkpoint',
                'async_scheduling': vllm_config.scheduler_config.async_scheduling,
                'graph_mode': str(vllm_config.compilation_config.cudagraph_mode)}
            (directory / 'cache-geometry.json').write_text(json.dumps(manifest, indent=2) + '\n')

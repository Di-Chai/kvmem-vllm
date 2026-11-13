"""Explicit capacity qualification for finite, immutable >1M MAB histories.

The benchmark ingress reserves whole histories with CpuCredits. Unlike an
unbounded rolling session, a finite input needs no deleting historical GC.
This option neither changes the dense K+G frame nor drops archived evidence.
"""
import json
from pathlib import Path


def validate_runtime_capacity(config, options):
    """Repeat the finite archive checks in native startup, before allocation."""
    model=config.model_config.hf_text_config
    model=model.to_dict() if hasattr(model,'to_dict') else vars(model)
    return validate_capacity(config.model_config.max_model_len,options,model)


def validate_full_history(cfg, options):
    memory=options.memory
    if not (cfg['model_format']=='safetensors' and cfg['draft_method']=='mtp'
            and cfg['max_model_len']<=2097152 and options.request_planning
            and options.archive_backend=='cpu' and options.cpu_history_coordination
            and options.cpu_progress_reserve and options.session_freeze_prompt
            and options.session_capacity and options.historical_gc is None
            and not options.repository_gc and not options.rolling_active_window
            and memory.select_budget+memory.gen_budget<=262144):
        raise ValueError('Full-history benchmark requires finite owned CPU histories and bounded MTP')
    model=json.loads((Path(cfg['hf_config_path'])/'config.json').read_text())
    return validate_capacity(cfg['max_model_len'],options,model.get('text_config',model))


def validate_capacity(limit, options, model):
    memory=options.memory
    if not (0<limit<=2097152 and options.archive_backend=='cpu'
            and options.session_freeze_prompt and options.session_capacity
            and options.historical_gc is None and not options.repository_gc
            and not options.rolling_active_window):
        raise ValueError('Full-history benchmark requires finite immutable CPU history')
    extent=limit+2*memory.gen_budget
    if memory.repository_blocks*memory.block_tokens<extent:
        raise ValueError('Full-history benchmark repository cannot retain the complete input')
    # This first qualification is deliberately limited to the existing Qwen
    # hybrid target with one full-attention MTP layer and FP8 immutable K/V.
    if model.get('model_type')!='qwen3_5_text' or 'layer_types' not in model:
        raise ValueError('Full-history benchmark model layout has not been qualified')
    layers=model['layer_types'].count('full_attention')+1
    required=extent*layers*model['num_key_value_heads']*model['head_dim']*2
    if options.cpu_read_cache_bytes-16*1024**3<required:
        raise ValueError('Full-history benchmark CPU credits cannot fit one complete history')
    return dict(full_history_bytes=required,logical_extent_tokens=extent,archive_layers=layers)

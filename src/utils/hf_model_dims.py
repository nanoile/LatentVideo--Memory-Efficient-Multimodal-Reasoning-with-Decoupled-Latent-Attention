"""Dimensions derived from HuggingFace configs (VL models vs flat LMs)."""


def inputs_embeds_hidden_size(config) -> int:
    """Last dim for `inputs_embeds` into the main forward (e.g. Qwen3-VL uses text_config.hidden_size)."""
    tc = getattr(config, "text_config", None)
    if tc is not None:
        h = getattr(tc, "hidden_size", None)
        if h is not None:
            return int(h)
    return int(config.hidden_size)


def set_model_use_cache_false(model) -> None:
    """Match gradient checkpointing: avoid KV cache allocation (VL models may nest text_config)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    tc = getattr(cfg, "text_config", None)
    if tc is not None and hasattr(tc, "use_cache"):
        tc.use_cache = False

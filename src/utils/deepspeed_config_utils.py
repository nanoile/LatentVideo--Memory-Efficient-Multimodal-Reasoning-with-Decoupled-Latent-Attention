"""DeepSpeed JSON hygiene (ZeRO pydantic rejects some keys nested under zero_optimization)."""

from __future__ import annotations

# Must be top-level keys in the DeepSpeed config, not inside zero_optimization
# (DeepSpeedZeroConfig has pydantic extra="forbid").
_ZERO_OPT_MISPLACED_KEYS = (
    "zero_allow_untested_optimizer",
    "zero_force_ds_cpu_optimizer",
)


def normalize_deepspeed_config_dict_inplace(ds_cfg: dict) -> None:
    """Move misplaced keys from zero_optimization to root; mutates ds_cfg."""
    zo = ds_cfg.get("zero_optimization")
    if not isinstance(zo, dict):
        return
    for k in _ZERO_OPT_MISPLACED_KEYS:
        if k in zo:
            ds_cfg[k] = zo.pop(k)

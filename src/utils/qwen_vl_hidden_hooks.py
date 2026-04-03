"""Capture specific Qwen3-VL text decoder layer outputs without output_hidden_states=True."""

from __future__ import annotations

from typing import Any, List, Tuple


def register_qwen3_vl_text_layer_hooks(
    causal_lm: Any, layers_to_match: List[int]
) -> Tuple[List[list], List[Any]]:
    """
    Register forward hooks on language_model.layers so we can omit output_hidden_states.

    Assumes teacher hidden_states[k] aligns with decoder layer output after layer (k - 1)
    when k > 0 (standard HF layout: hidden_states[0] after embed, then after each block).
    """
    lm = causal_lm.model.language_model
    layers = lm.layers
    n = len(layers)
    bufs: List[list] = [[] for _ in layers_to_match]
    handles = []
    for buf, hs_idx in zip(bufs, layers_to_match):
        li = int(hs_idx) - 1
        li = max(0, min(li, n - 1))

        def _hook(_m, _inp, out, b=buf):
            b.clear()
            b.append(out)

        handles.append(layers[li].register_forward_hook(_hook))
    return bufs, handles

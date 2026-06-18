"""Register the NemotronH (Mamba2 hybrid) packed-doc reset as a post-load-lifetime patch.

The kernel logic lives in ``models/nemotron_h.py`` (resets Mamba2 conv+scan via seq_idx on the
un-fused branch AND attention via flash_attn_varlen_func with per-doc cu_seqlens — both must reset
together; resetting only one is worse than neither). It needs the instantiated model to locate the
remote-modeling Mamba2 mixer + attention classes, so it registers at the ``post_load`` lifetime.
"""

from ..registry import PackingPatch, register_packing_patch


def _applies(hf_config) -> bool:
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _apply(model):
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


register_packing_patch(PackingPatch("nemotron_h_packing", _applies, "post_load", _apply))

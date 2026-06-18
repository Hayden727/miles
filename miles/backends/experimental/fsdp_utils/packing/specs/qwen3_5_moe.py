"""Register the GatedDeltaNet (Qwen3.5/3.6, Qwen3-Next) packed-doc reset as a config-lifetime patch.

The kernel logic lives in ``models/qwen3_5_moe.py`` (class-forward patches that feed cu_seqlens to
fla chunk/recurrent_gated_delta_rule and seq_idx to causal_conv1d_fn per packed document). This spec
only wires that proven patch into the unified packing registry; ``applies_to`` reuses the existing
``_is_gated_deltanet`` predicate (imported lazily to avoid an import cycle).
"""

from ..registry import PackingPatch, register_packing_patch


def _applies(hf_config) -> bool:
    if hf_config is None:
        return False
    from ...hf_compat_patches import _is_gated_deltanet

    return _is_gated_deltanet(hf_config)


def _apply():
    from ...models.qwen3_5_moe import apply_gateddeltanet_packing_patch

    return apply_gateddeltanet_packing_patch()


register_packing_patch(PackingPatch("gated_deltanet_packing", _applies, "config", _apply))

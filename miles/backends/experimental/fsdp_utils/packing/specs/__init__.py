"""Per-arch packing specs. Importing this package registers every arch's PackingPatch.

A new packing arch adds a module here that calls ``register_packing_patch(...)`` and an import
below. Archs whose attention/state already handles packing natively (e.g. glm4_moe_lite's MLA via
HF FlashAttentionKwargs) intentionally register nothing.
"""

from . import nemotron_h, qwen3_5_moe  # noqa: F401  (import triggers registration)

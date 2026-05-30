"""Local Qwen3-VL Bridge shims for Miles THD packed batches."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


_PATCHED_ATTR = "_miles_qwen_vl_packed_mrope_patch"


def install_qwen_vl_packed_mrope_patch() -> None:
    """Install a local Megatron Bridge Qwen3-VL packed mRoPE patch if available."""

    _patch_rotary_signature()
    _patch_qwen_vl_models()


def _patch_rotary_signature() -> None:
    try:
        text_model = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model")
    except ImportError:
        return

    for name in ("Qwen3VLTextRotaryEmbedding", "Qwen3VLMoETextRotaryEmbedding"):
        cls = getattr(text_model, name, None)
        if cls is None or cls.__dict__.get(_PATCHED_ATTR, False):
            continue
        cls.forward = _make_rotary_forward(cls.forward)
        setattr(cls, _PATCHED_ATTR, True)


def _patch_qwen_vl_models() -> None:
    for module_name in (
        "megatron.bridge.models.qwen_vl.modeling_qwen3_vl",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.modeling_qwen3_vl",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.moe_model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model_moe",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        for cls in _iter_qwen_vl_model_classes(module):
            _patch_model_forward(cls)


def _make_rotary_forward(original_forward):
    def patched_forward(self, *args, **kwargs):
        kwargs.pop("packed_seq_params", None)
        return original_forward(self, *args, **kwargs)

    return patched_forward


def _iter_qwen_vl_model_classes(module: Any):
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if "Qwen3VL" not in name:
            continue
        if not hasattr(obj, "forward") or not hasattr(obj, "get_rope_index"):
            continue
        yield obj


def _patch_model_forward(cls: type) -> None:
    if cls.__dict__.get(_PATCHED_ATTR, False):
        return

    original_forward = cls.forward

    def patched_forward(self, *args, **kwargs):
        if kwargs.get("position_ids") is None:
            position_ids, rope_deltas = _try_build_packed_mrope_position_ids(
                self,
                input_ids=kwargs.get("input_ids"),
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"),
                packed_seq_params=kwargs.get("packed_seq_params"),
            )
            if rope_deltas is not None and hasattr(self, "rope_deltas"):
                self.rope_deltas = rope_deltas
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

        return original_forward(self, *args, **kwargs)

    cls.forward = patched_forward
    setattr(cls, _PATCHED_ATTR, True)


def _try_build_packed_mrope_position_ids(
    model: Any,
    *,
    input_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    packed_seq_params: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if input_ids is None or packed_seq_params is None:
        return None, None
    if getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None, None
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        return None, None
    if not hasattr(model, "get_rope_index"):
        return None, None

    cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
    if cu_seqlens is None:
        return None, None

    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    flat_input_ids = input_ids.squeeze(0)
    if not cu or cu[0] != 0 or cu[-1] != flat_input_ids.numel():
        logger.debug(
            "Skipping Qwen3-VL packed mRoPE patch because cu_seqlens=%s does not match input length=%s",
            cu,
            flat_input_ids.numel(),
        )
        return None, None

    image_offset = 0
    video_offset = 0
    packed_position_ids: list[torch.Tensor] = []
    rope_deltas: list[torch.Tensor] = []

    for start, end in zip(cu[:-1], cu[1:], strict=False):
        segment = flat_input_ids[start:end]
        if segment.numel() == 0:
            continue

        image_count, video_count = _count_segment_media(model, segment)
        segment_image_grid = _slice_optional_grid(image_grid_thw, image_offset, image_count)
        segment_video_grid = _slice_optional_grid(video_grid_thw, video_offset, video_count)
        image_offset += image_count
        video_offset += video_count

        if image_count == 0 and video_count == 0:
            pos, delta = _linear_position_ids(segment)
        else:
            pos, delta = model.get_rope_index(
                input_ids=segment.unsqueeze(0),
                image_grid_thw=segment_image_grid,
                video_grid_thw=segment_video_grid,
                attention_mask=torch.ones((1, segment.numel()), dtype=torch.long, device=segment.device),
            )

        packed_position_ids.append(pos[:, 0, : segment.numel()])
        rope_deltas.append(delta.reshape(1, -1))

    if not packed_position_ids:
        return None, None

    return torch.cat(packed_position_ids, dim=1).unsqueeze(1), torch.cat(rope_deltas, dim=0)


def _count_segment_media(model: Any, segment: torch.Tensor) -> tuple[int, int]:
    config = getattr(model, "config", model)
    vision_start_token_id = getattr(config, "vision_start_token_id", None)
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)
    if vision_start_token_id is None or image_token_id is None or video_token_id is None:
        return 0, 0

    starts = torch.nonzero(segment == vision_start_token_id, as_tuple=False).flatten()
    starts = starts[starts + 1 < segment.numel()]
    if starts.numel() == 0:
        return 0, 0

    vision_tokens = segment[starts + 1]
    return int((vision_tokens == image_token_id).sum().item()), int((vision_tokens == video_token_id).sum().item())


def _slice_optional_grid(grid: torch.Tensor | None, offset: int, count: int) -> torch.Tensor | None:
    if grid is None or count == 0:
        return None
    return grid[offset : offset + count]


def _linear_position_ids(segment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = (
        torch.arange(segment.numel(), dtype=segment.dtype, device=segment.device).view(1, 1, -1).expand(3, 1, -1)
    )
    rope_delta = torch.zeros((1, 1), dtype=segment.dtype, device=segment.device)
    return position_ids, rope_delta

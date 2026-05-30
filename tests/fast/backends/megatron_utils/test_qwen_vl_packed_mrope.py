from __future__ import annotations

import sys
import types
from importlib import util
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "miles" / "backends" / "megatron_utils" / "qwen_vl_packed_mrope.py"
)
_SPEC = util.spec_from_file_location("qwen_vl_packed_mrope_for_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qwen_vl_packed_mrope = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qwen_vl_packed_mrope)

_try_build_packed_mrope_position_ids = qwen_vl_packed_mrope._try_build_packed_mrope_position_ids
install_qwen_vl_packed_mrope_patch = qwen_vl_packed_mrope.install_qwen_vl_packed_mrope_patch


class PackedSeqParams:
    qkv_format = "thd"

    def __init__(self, cu_seqlens):
        self.cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int)


class FakeConfig:
    vision_start_token_id = 10
    image_token_id = 11
    video_token_id = 12


class FakeQwen3VLModel:
    config = FakeConfig()

    def __init__(self):
        self.calls = []
        self.rope_deltas = None

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        image_count = 0 if image_grid_thw is None else image_grid_thw.size(0)
        video_count = 0 if video_grid_thw is None else video_grid_thw.size(0)
        self.calls.append((input_ids.clone(), image_count, video_count))
        base = 100 * len(self.calls)
        position_ids = torch.arange(base, base + input_ids.size(1), dtype=input_ids.dtype).view(1, 1, -1)
        return position_ids.expand(3, 1, -1), torch.tensor([[image_count + video_count]], dtype=input_ids.dtype)

    def forward(self, **kwargs):
        return kwargs


def test_builds_packed_mrope_positions_per_segment():
    model = FakeQwen3VLModel()
    input_ids = torch.tensor([[1, 10, 11, 2, 3, 4]])
    packed_seq_params = PackedSeqParams([0, 4, 6])
    image_grid_thw = torch.tensor([[1, 14, 14]])

    position_ids, rope_deltas = _try_build_packed_mrope_position_ids(
        model,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        packed_seq_params=packed_seq_params,
    )

    assert position_ids.shape == (3, 1, 6)
    assert position_ids[0, 0].tolist() == [100, 101, 102, 103, 0, 1]
    assert rope_deltas.tolist() == [[1], [0]]
    assert len(model.calls) == 1
    assert model.calls[0][0].tolist() == [[1, 10, 11, 2]]
    assert model.calls[0][1:] == (1, 0)


def test_skips_when_cu_seqlens_do_not_match_local_input():
    model = FakeQwen3VLModel()
    input_ids = torch.tensor([[1, 10, 11, 2]])
    packed_seq_params = PackedSeqParams([0, 8])

    position_ids, rope_deltas = _try_build_packed_mrope_position_ids(
        model,
        input_ids=input_ids,
        image_grid_thw=torch.tensor([[1, 14, 14]]),
        video_grid_thw=None,
        packed_seq_params=packed_seq_params,
    )

    assert position_ids is None
    assert rope_deltas is None
    assert model.calls == []


def test_install_patch_supplies_position_ids_to_fake_bridge_model(monkeypatch):
    _install_fake_bridge_modules(monkeypatch, FakeQwen3VLModel)

    install_qwen_vl_packed_mrope_patch()

    model = FakeQwen3VLModel()
    result = model.forward(
        input_ids=torch.tensor([[1, 10, 11, 2]]),
        position_ids=None,
        image_grid_thw=torch.tensor([[1, 14, 14]]),
        packed_seq_params=PackedSeqParams([0, 4]),
    )

    assert result["position_ids"].shape == (3, 1, 4)
    assert result["position_ids"][0, 0].tolist() == [100, 101, 102, 103]
    assert model.rope_deltas.tolist() == [[1]]


def _install_fake_bridge_modules(monkeypatch, model_cls):
    text_model = types.ModuleType("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model")
    text_model.Qwen3VLTextRotaryEmbedding = type("Qwen3VLTextRotaryEmbedding", (), {"forward": lambda self, x: x})
    text_model.Qwen3VLMoETextRotaryEmbedding = type(
        "Qwen3VLMoETextRotaryEmbedding", (), {"forward": lambda self, x: x}
    )

    model_module = types.ModuleType("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    model_module.Qwen3VLModel = model_cls

    for name in (
        "megatron",
        "megatron.bridge",
        "megatron.bridge.models",
        "megatron.bridge.models.qwen_vl",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, text_model.__name__, text_model)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

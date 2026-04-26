"""Unit tests for ``resolve_fixed_chat_template`` table lookup."""

import os

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, resolve_fixed_chat_template

_QWEN3_FIXED = str(TEMPLATE_DIR / "qwen3_fixed.jinja")
_QWEN35_FIXED = str(TEMPLATE_DIR / "qwen3.5_fixed.jinja")
_THINKING_FIXED = str(TEMPLATE_DIR / "qwen3_thinking_2507_and_next_fixed.jinja")


@pytest.mark.parametrize(
    "tito_model, expected_path",
    [
        (TITOTokenizerType.QWEN3, _QWEN3_FIXED),
        (TITOTokenizerType.QWEN35, _QWEN35_FIXED),
        (TITOTokenizerType.QWENNEXT, _THINKING_FIXED),
    ],
)
def test_tool_only_resolution_matches_registered_template(tito_model, expected_path):
    path = resolve_fixed_chat_template(tito_model, ["tool"])
    assert path == expected_path
    assert os.path.isfile(path)


def test_unregistered_role_set_returns_none():
    assert resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool", "user"]) is None


def test_unregistered_tito_model_returns_none():
    assert resolve_fixed_chat_template(TITOTokenizerType.GLM47, ["tool"]) is None

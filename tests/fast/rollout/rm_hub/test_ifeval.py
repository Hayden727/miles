"""Tests for the IFEval reward wrapper (miles.rollout.rm_hub.ifeval).

Two layers:

* Pure-helper and fake-lib tests run everywhere with no external deps — they pin
  down our metadata parsing and the strict/loose routing that this module adds.
* The ``TestComputeIfevalRewardOfficial`` cases drive the real, official Google
  IFEval ``evaluation_lib`` end to end. They are skipped unless its pip deps are
  installed (fast CI legitimately lacks them), so the cheap layers above are what
  actually guard the wrapper in CI; the official layer is the upper-bound check.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, ifeval
from miles.utils.async_utils import run
from miles.utils.types import Sample


class TestNormalizeInstructionIds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["keywords:existence", "punctuation:no_comma"], ["keywords:existence", "punctuation:no_comma"]),
            (
                [" keywords:existence ", "", None, "punctuation:no_comma"],
                ["keywords:existence", "punctuation:no_comma"],
            ),
            ([1, 2], ["1", "2"]),
            ([], []),
            (None, []),
        ],
    )
    def test_normalize(self, raw, expected):
        assert ifeval._normalize_instruction_ids(raw) == expected


class TestCoerceKwargsList:
    def test_list_passthrough(self):
        assert ifeval._coerce_kwargs_list([{"keywords": ["a"]}], 1) == [{"keywords": ["a"]}]

    def test_dict_broadcast(self):
        assert ifeval._coerce_kwargs_list({"num_words": 3}, 2) == [{"num_words": 3}, {"num_words": 3}]

    def test_none_or_other_yields_empty_dicts(self):
        assert ifeval._coerce_kwargs_list(None, 2) == [{}, {}]
        assert ifeval._coerce_kwargs_list("nonsense", 1) == [{}]

    def test_non_dict_entries_become_empty(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}, "bad"], 2) == [{"a": 1}, {}]

    def test_padding_repeats_tail(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}], 3) == [{"a": 1}, {"a": 1}, {"a": 1}]

    def test_truncation(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}, {"b": 2}, {"c": 3}], 2) == [{"a": 1}, {"b": 2}]

    def test_none_values_dropped(self):
        assert ifeval._coerce_kwargs_list([{"keep": 1, "drop": None}], 1) == [{"keep": 1}]


@dataclass
class _FakeInputExample:
    key: int
    instruction_id_list: list
    prompt: str
    kwargs: list


class _FakeOutput:
    def __init__(self, follow_all_instructions):
        self.follow_all_instructions = follow_all_instructions


class _FakeLib:
    """Stand-in for instruction_following_eval.evaluation_lib.

    Records which criterion was invoked and the InputExample it was handed, so the
    wrapper tests can assert routing and metadata translation without the real lib.
    """

    def __init__(self, follow_strict, follow_loose):
        self.InputExample = _FakeInputExample
        self._follow_strict = follow_strict
        self._follow_loose = follow_loose
        self.calls: list[str] = []
        self.last_inp = None
        self.last_prompt_to_response = None

    def test_instruction_following_strict(self, inp, prompt_to_response):
        self.calls.append("strict")
        self.last_inp = inp
        self.last_prompt_to_response = prompt_to_response
        return _FakeOutput(self._follow_strict)

    def test_instruction_following_loose(self, inp, prompt_to_response):
        self.calls.append("loose")
        self.last_inp = inp
        self.last_prompt_to_response = prompt_to_response
        return _FakeOutput(self._follow_loose)


@pytest.fixture
def install_fake_lib(monkeypatch):
    """Inject a fake evaluation_lib so compute_ifeval_reward never clones/imports."""

    def _install(follow_strict=False, follow_loose=False):
        lib = _FakeLib(follow_strict=follow_strict, follow_loose=follow_loose)
        monkeypatch.setattr(ifeval, "_evaluation_lib", lib)
        return lib

    return _install


class TestComputeIfevalRewardWrapper:
    def test_none_metadata_returns_zero(self, install_fake_lib):
        install_fake_lib(follow_strict=True, follow_loose=True)
        assert ifeval.compute_ifeval_reward("anything", None, metadata=None) == 0.0

    def test_none_response_returns_zero(self, install_fake_lib):
        install_fake_lib(follow_strict=True, follow_loose=True)
        metadata = {"instruction_id_list": ["keywords:existence"]}
        assert ifeval.compute_ifeval_reward(None, None, metadata=metadata) == 0.0

    def test_missing_instruction_ids_returns_zero(self, install_fake_lib):
        install_fake_lib(follow_strict=True, follow_loose=True)
        assert ifeval.compute_ifeval_reward("resp", None, metadata={"instruction_id_list": []}) == 0.0

    def test_default_mode_is_strict(self, install_fake_lib):
        lib = install_fake_lib(follow_strict=True, follow_loose=False)
        reward = ifeval.compute_ifeval_reward("resp", None, metadata={"instruction_id_list": ["keywords:existence"]})
        assert reward == 1.0
        assert lib.calls == ["strict"]

    def test_loose_mode_routes_to_loose(self, install_fake_lib):
        lib = install_fake_lib(follow_strict=False, follow_loose=True)
        reward = ifeval.compute_ifeval_reward(
            "resp", None, metadata={"instruction_id_list": ["keywords:existence"]}, strict=False
        )
        assert reward == 1.0
        assert lib.calls == ["loose"]

    def test_not_following_returns_zero(self, install_fake_lib):
        install_fake_lib(follow_strict=False, follow_loose=False)
        reward = ifeval.compute_ifeval_reward("resp", None, metadata={"instruction_id_list": ["keywords:existence"]})
        assert reward == 0.0

    def test_input_example_built_from_metadata(self, install_fake_lib):
        lib = install_fake_lib(follow_strict=True, follow_loose=True)
        metadata = {
            "instruction_id_list": [" keywords:existence "],
            "kwargs": [{"keywords": ["foo"], "num_words": None}],
            "prompt_text": "Write something.",
            "record_id": 7,
        }
        ifeval.compute_ifeval_reward("my response", None, metadata=metadata)
        assert lib.last_inp.key == 7
        assert lib.last_inp.instruction_id_list == ["keywords:existence"]
        # None-valued kwargs are dropped so build_description only sees declared keys.
        assert lib.last_inp.kwargs == [{"keywords": ["foo"]}]
        assert lib.last_inp.prompt == "Write something."
        assert lib.last_prompt_to_response == {"Write something.": "my response"}


# --- Integration against the official Google IFEval library -------------------

_IFEVAL_DEP_MODULES = ("absl", "immutabledict", "langdetect", "nltk")


@functools.lru_cache(maxsize=1)
def _official_lib_available() -> bool:
    # Cheap gate first: if the pip deps are not installed, do not attempt a network
    # checkout. Only once they are present do we let the production loader fetch and
    # import the source (raising → unavailable rather than failing the suite).
    if any(importlib.util.find_spec(name) is None for name in _IFEVAL_DEP_MODULES):
        return False
    try:
        ifeval._load_evaluation_lib()
        return True
    except Exception:
        return False


requires_official_ifeval = pytest.mark.skipif(
    not _official_lib_available(),
    reason="official Google IFEval source/deps not available in this environment",
)


@requires_official_ifeval
class TestComputeIfevalRewardOfficial:
    """End-to-end checks against the real official checkers.

    Cases below are chosen so that strict and loose agree (single-line, no markdown)
    except the final mode-sensitive case, which only passes once loose strips the
    offending first line — that one pins down that the strict/loose routing is real.
    All chosen instructions are regex/string based (no nltk punkt data, no langdetect),
    so results are deterministic.
    """

    @pytest.mark.parametrize(
        "instruction_id,kwargs,response,expected_strict,expected_loose",
        [
            # keyword existence: both required keywords present vs. one missing
            (
                "keywords:existence",
                {"keywords": ["banana", "umbrella"]},
                "I bought a banana and an umbrella today.",
                1.0,
                1.0,
            ),
            ("keywords:existence", {"keywords": ["banana", "umbrella"]}, "I bought a banana today.", 0.0, 0.0),
            # no comma allowed
            ("punctuation:no_comma", {}, "I have no commas here", 1.0, 1.0),
            ("punctuation:no_comma", {}, "I have a comma, right here", 0.0, 0.0),
            # response must end with an exact phrase
            ("startend:end_checker", {"end_phrase": "That is all."}, "Here is my answer. That is all.", 1.0, 1.0),
            ("startend:end_checker", {"end_phrase": "That is all."}, "Here is my answer.", 0.0, 0.0),
            # word count: count_words uses a regex tokenizer, so no punkt data needed
            (
                "length_constraints:number_words",
                {"num_words": 5, "relation": "at least"},
                "one two three four five six",
                1.0,
                1.0,
            ),
            ("length_constraints:number_words", {"num_words": 5, "relation": "at least"}, "one two three", 0.0, 0.0),
            # mode-sensitive: the comma sits on the first line, so loose passes after
            # stripping it while strict still sees it.
            ("punctuation:no_comma", {}, "Listen, this is the intro.\nThis line is clean", 0.0, 1.0),
        ],
    )
    def test_reward_matches_official(self, instruction_id, kwargs, response, expected_strict, expected_loose):
        metadata = {
            "instruction_id_list": [instruction_id],
            "kwargs": [kwargs],
            "prompt_text": "Follow the instruction.",
            "record_id": 0,
        }
        assert ifeval.compute_ifeval_reward(response, None, metadata=metadata, strict=True) == expected_strict
        assert ifeval.compute_ifeval_reward(response, None, metadata=metadata, strict=False) == expected_loose


class TestIfevalRmTypeDispatch:
    """The rm_type token must select the right criterion: bare ``ifeval`` and
    ``ifeval_strict`` route to strict, ``ifeval_loose`` to loose. A fake lib is
    injected so this exercises the full async_rm dispatch without a network clone."""

    @pytest.mark.parametrize(
        "rm_type,expected_call",
        [
            ("ifeval", "strict"),
            ("ifeval_strict", "strict"),
            ("ifeval_loose", "loose"),
        ],
    )
    def test_rm_type_routes_to_mode(self, monkeypatch, rm_type, expected_call):
        lib = _FakeLib(follow_strict=True, follow_loose=True)
        monkeypatch.setattr(ifeval, "_evaluation_lib", lib)

        args = MagicMock()
        args.custom_rm_path = None
        args.rm_type = rm_type
        sample = Sample(
            prompt="",
            response="resp",
            label=None,
            metadata={"instruction_id_list": ["keywords:existence"]},
        )
        reward = run(async_rm(args, sample))

        assert reward == 1.0
        assert lib.calls == [expected_call]

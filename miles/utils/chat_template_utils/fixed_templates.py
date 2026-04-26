"""Resolve repo-bundled fixed chat templates from ``(tito_model, allowed_append_roles)``.

The single public entry point :func:`resolve_fixed_chat_template` takes the TITO
tokenizer family that determines token-level merge behavior — and therefore
which jinja file is correct — together with the append-role surface the caller
intends to use.  A missing entry resolves to ``None`` (caller falls back to HF
default).  See ``docs/en/agentic/tito_validation_refactor_plan.zh.md`` for the
broader design context.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


# ``tool`` is the implicit baseline for any agentic workflow — appending tool
# results never breaks the append-only invariant — so it is unioned in below.
_IMPLICIT_APPEND_ROLES: frozenset[str] = frozenset({"tool"})


# Sparse ``(tito_model, normalized-role-set) -> filename`` table.  Missing
# entries mean "no fixed template registered for this combination" and resolve
# to ``None`` (caller falls back to HF default).  Add a row when registering
# a new verified template.
_FIX_TEMPLATE: dict[tuple[TITOTokenizerType, frozenset[str]], str] = {
    (TITOTokenizerType.QWEN3, frozenset({"tool"})): "qwen3_fixed.jinja",
    (TITOTokenizerType.QWEN35, frozenset({"tool"})): "qwen3.5_fixed.jinja",
    (TITOTokenizerType.QWENNEXT, frozenset({"tool"})): "qwen3_thinking_2507_and_next_fixed.jinja",
}


def resolve_fixed_chat_template(
    tito_model: TITOTokenizerType | str,
    allowed_append_roles: Iterable[str] | None = None,
) -> str | None:
    """Look up a bundled fixed chat-template path for ``(tito_model, roles)``.

    *allowed_append_roles* is lowercased, deduped, and unioned with the
    implicit ``tool`` baseline before the lookup.  Returns ``None`` when no
    template is registered for the resulting combination — callers should
    fall back to the HF default in that case.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)

    raw = {r.lower() for r in (allowed_append_roles or [])}
    roles = frozenset(raw | _IMPLICIT_APPEND_ROLES)

    filename = _FIX_TEMPLATE.get((tito_model, roles))
    if filename is None:
        return None

    path = str(TEMPLATE_DIR / filename)
    logger.info(
        "tito_model=%s roles=%s -> using fixed chat template %s",
        tito_model.value,
        sorted(roles),
        path,
    )
    return path

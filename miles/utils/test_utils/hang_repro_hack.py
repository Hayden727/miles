"""HACK ft-hang-repro: deterministic reproduction of the update_weights peer-death wedge.

This module exists ONLY to reproduce, on demand and every time, the bug where a cell that
dies inside ``update_weights`` leaves a surviving cell wedged in its own (unguarded)
``update_weights`` collective. It is a throwaway debugging aid; revert it via
``git grep 'HACK ft-hang-repro'`` once the root cause is understood.

Enable by exporting ``MILES_FT_HACK_KILL_AT_UPDATE_WEIGHTS=<N>``: on the Nth ``update_weights``
call on rank 0 of the active source cell, the process segfaults (matching the soak's segfault
fault mode), abruptly killing the cell mid-weight-update. Because ``update_weights`` only runs
on the first-alive cell, the counter naturally tracks that cell, and the controller's retry
promotes a survivor cell whose own ``update_weights`` is the thing we want to observe wedge.
"""

import logging
import os

import torch.distributed as dist

from miles.utils.test_utils.fault_injector import inject_fault

logger = logging.getLogger(__name__)

_ENV_KILL_AT_UPDATE_WEIGHTS = "MILES_FT_HACK_KILL_AT_UPDATE_WEIGHTS"

_update_weights_call_count = 0


def maybe_kill_at_update_weights(phase: str) -> None:
    global _update_weights_call_count

    raw_target = os.environ.get(_ENV_KILL_AT_UPDATE_WEIGHTS)
    if not raw_target:
        return
    if not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
        return

    _update_weights_call_count += 1
    if _update_weights_call_count != int(raw_target):
        return

    logger.warning(
        "HACK ft-hang-repro: deterministic segfault at update_weights phase=%s call=%d pid=%d",
        phase,
        _update_weights_call_count,
        os.getpid(),
    )
    inject_fault("segfault")

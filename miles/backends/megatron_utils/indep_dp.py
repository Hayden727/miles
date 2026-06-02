import logging
import threading
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GeneralPGUtil, GroupInfo, collective_bool_and

from ..training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=120)

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        pg = pg_cls(timeout=_TIMEOUT)
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
            replica_id=str(indep_dp_info.cell_index),
            rank=indep_dp_info.alive_rank,
            world_size=indep_dp_info.alive_size,
            quorum_id=indep_dp_info.quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    logger.info(
        f"Configured independent DP PG: {indep_dp_info}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}"
    )
    return GroupInfo(rank=indep_dp_info.alive_rank, size=indep_dp_info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def _shutdown_pg_off_critical_path(g, timeout_s: float = 20.0) -> None:
    """Shut down an old indep_dp PG without letting a dead peer block recovery.

    torchft ``ProcessGroup.shutdown`` drops the last ref and runs
    ``~ProcessGroupNCCL``, whose ``abort()`` calls ``waitForFutureOrTimeout`` —
    that blocks forever when a peer cell died with a collective still in flight
    (NCCL cannot drain it over NVLink). Only ranks whose comm already hit the
    torchft watchdog abort shut down instantly; the rest would hang here and
    stall the surviving cell's reconfigure. Run shutdown on a daemon thread so a
    stuck teardown is abandoned instead of blocking; the orphaned comm is
    harmless because we immediately build a fresh quorum.
    """
    t = threading.Thread(target=g.shutdown, name="indep-dp-pg-shutdown", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            "indep_dp old PG shutdown still blocked after %ss; abandoning it (peer cell likely dead)",
            timeout_s,
        )


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> None:
    """Shutdown old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    for g in [old.group, old.gloo_group]:
        if g is not None:
            _shutdown_pg_off_critical_path(g)

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    logger.info(f"Reconfigured indep_dp PG with quorum_id={indep_dp_info.quorum_id}")


def _exact_sum_across_replicas(
    util: GeneralPGUtil,
    grad_data: torch.Tensor,
    pg: dist.ProcessGroup,
    *,
    chunk_numel: int = 64 * 1024 * 1024,
) -> None:
    """Sum grad_data across alive cells in ascending alive_rank order (bitwise-stable).

    NCCL all_reduce does not guarantee a bitwise-identical reduction order across
    comm instances, so a recovered cross-cell collective (rebuilt with a new
    quorum_id) can sign-flip near-zero cancellation residuals versus the no-fault
    baseline. Under --deterministic-mode we instead all_gather every cell's
    contribution and sum them in a fixed rank order that is identical in the
    baseline and the recovered run, making the reduction itself bitwise-stable.

    Chunked so peak extra memory is ``chunk_numel * world`` rather than
    ``world * full_grad`` (an unchunked all_gather OOMs on large expert buckets).
    Goes through ``util.all_gather`` -> ``_check_wait``, so a dead peer raises
    here exactly like the all_reduce path and is handled by the caller's consensus.
    """
    world = util.get_size(pg)
    flat = grad_data.view(-1)
    total = flat.numel()
    for start in range(0, total, chunk_numel):
        chunk = flat[start : start + chunk_numel]
        gathered = [torch.empty_like(chunk) for _ in range(world)]
        util.all_gather(gathered, chunk, pg)
        acc = gathered[0].clone()
        for other in gathered[1:]:
            acc += other
        chunk.copy_(acc)


def _allreduce_grads_across_replicas(args, model: Sequence["DDP"], parallel_state: ParallelState) -> bool:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    pg = parallel_state.indep_dp.group
    util = GeneralPGUtil.create(pg)
    deterministic = bool(args.deterministic_mode)

    allreduce_success = True
    try:
        for model_chunk in model:
            # mimic: DistributedDataParallel.start_grad_sync
            for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
                for bucket in bucket_group.buckets:
                    if deterministic:
                        _exact_sum_across_replicas(util, bucket.grad_data, pg)
                    else:
                        util.all_reduce(bucket.grad_data, pg, op=dist.ReduceOp.SUM)
    except Exception:
        allreduce_success = False
        logger.exception("Gradient allreduce across replicas failed")

    if (e := pg.errored()) is not None:
        allreduce_success = False
        logger.error("indep_dp PG has async error: %s", e)

    # Intra-cell consensus: if ANY rank's allreduce failed, ALL ranks discard.
    # get_gloo_group() is cell-local (created from the default world PG).
    return collective_bool_and(value=allreduce_success, group=get_gloo_group())

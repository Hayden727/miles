"""Reference for correct torchft ProcessGroup usage in miles (incl. teardown/recovery).

NOT a CI test (needs a Ray GPU cluster, so it is intentionally not a ``test_*.py`` and
not registered). It is the canonical, *runnable* reference for how miles uses torchft's
``ProcessGroupNCCL`` / ``ProcessGroupGloo`` — mirroring
``miles/backends/megatron_utils/indep_dp.py`` (``create_indep_dp_group`` /
``reconfigure_indep_dp_group``). Run it to (a) copy the correct API usage from, and
(b) confirm pure torchft behaves correctly — in particular that tearing a PG down after a
peer dies mid-collective does NOT hang (the property indep_dp recovery relies on).

Everything is asserted; the script exits 0 only if all of the following hold:
  1. Build a PG the correct way: a FRESH ``ProcessGroupNCCL(timeout=...)`` wrapper per
     quorum + ``configure(store_addr, replica_id, rank, world_size, quorum_id)``.
     (Do NOT reuse one wrapper across quorums — that is the torchft ``Manager`` pattern,
     and its leftover stream-timeout callback aborts the next quorum's comm.)
  2. Run a collective the Manager way: ``pg.allreduce([t], opts)`` -> ``work.wait()``,
     with ``pg.errored()`` checked, and get the correct numeric result.
  3. Recover from a dead peer: the survivor's failed ``wait()`` fires torchft's userspace
     abort during the bad collective; then tear the old NCCL+Gloo PGs down with a plain
     ``shutdown()`` and build a fresh wrapper for the new (singleton) quorum — exactly
     ``reconfigure_indep_dp_group``. This completes in seconds, no hang.

Usage:
    python tests/e2e/external/torchft_process_group_reference.py
    python tests/e2e/external/torchft_process_group_reference.py --tensor-size 200000000
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class _PGWorker:
    """Ray actor (1 GPU) that holds a torchft PG and uses it the way torchft Manager does."""

    def build_group(
        self,
        *,
        store_addr_q0: str,
        store_addr_q1: str,
        rank: int,
        world_size: int,
        timeout_s: float,
    ) -> dict:
        """Build the quorum-0 PG. Mirrors ``create_indep_dp_group``: a fresh NCCL (+Gloo)
        wrapper + ``configure``. ``store_addr_q1`` is kept for the later reconfigure."""
        import torch
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._rank = rank
        self._world_size = world_size
        self._timeout_s = timeout_s
        self._store_addr_q1 = store_addr_q1
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        self._pg = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._pg.configure(
            store_addr=store_addr_q0,
            replica_id=str(rank),
            rank=rank,
            world_size=world_size,
            quorum_id=0,
        )

        # miles builds a Gloo PG alongside NCCL and tears both down in reconfigure.
        self._gloo_pg = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
        self._gloo_pg.configure(
            store_addr=f"{store_addr_q0}/gloo",
            replica_id=str(rank),
            rank=rank,
            world_size=world_size,
            quorum_id=0,
        )

        return {"rank": rank, "nccl_version": torch.cuda.nccl.version(), "use_abort": self._pg._use_abort}

    def allreduce_check(self, *, expected: float) -> dict:
        """One Manager-style allreduce (SUM) on the current PG; assert the result."""
        import torch
        import torch.distributed as dist

        assert self._pg.errored() is None, f"rank {self._rank}: PG already errored"
        tensor = torch.ones(8, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        ok = self._pg.allreduce([tensor], opts).wait()
        value = tensor[0].item()
        assert ok and abs(value - expected) < 1e-6, f"rank {self._rank}: ok={ok} value={value} expected={expected}"
        return {"rank": self._rank, "ok": bool(ok), "value": value}

    def run_allreduce_then_die(self, *, tensor_size: int, die_after_s: float) -> None:
        """Start continuous allreduce, then ``os._exit`` mid-collective (the dead peer)."""
        import threading

        import torch
        import torch.distributed as dist

        def _delayed_exit() -> None:
            time.sleep(die_after_s)
            logger.warning("rank %d: os._exit after %.1fs (mid-allreduce kill)", self._rank, die_after_s)
            os._exit(1)

        threading.Thread(target=_delayed_exit, daemon=True).start()

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        while True:
            work = self._pg.allreduce([tensor], opts)
            work.wait()

    def survivor_detect(self, *, tensor_size: int, max_iters: int = 100_000) -> dict:
        """Loop allreduce until the peer's death is detected (the torchft-idiomatic path).

        ``ProcessGroupNCCL.allreduce`` returns a ``_WorkAcceleratorTimeout`` whose
        ``wait()`` arms a userspace ``context_timeout`` that fires ``pg.abort()`` after
        ``timeout_s``. So the in-flight comm is aborted *here*, during the failed wait —
        not at teardown. This is why the subsequent reconfigure is fast.
        """
        import torch
        import torch.distributed as dist

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM

        start = time.monotonic()
        count = 0
        status = "still_running"
        err = None
        try:
            for _ in range(max_iters):
                work = self._pg.allreduce([tensor], opts)
                if not work.wait():
                    status = "wait_false"
                    break
                count += 1
            else:
                status = "exhausted_iters"
        except Exception as e:
            status = "exception"
            err = f"{type(e).__name__}: {e}"

        elapsed = time.monotonic() - start
        try:
            errored_after = str(self._pg.errored())
        except Exception as e:
            errored_after = f"errored() raised {type(e).__name__}: {e}"

        return {
            "rank": self._rank,
            "status": status,
            "count": count,
            "error": err,
            "detect_elapsed_s": round(elapsed, 2),
            "errored_after": errored_after,
        }

    def reconfigure_to_singleton(self) -> dict:
        """Tear the dead-peer PGs down and build a fresh singleton quorum.

        Exactly ``reconfigure_indep_dp_group``: ``shutdown()`` the old NCCL + Gloo PGs,
        then build a fresh wrapper for the new quorum. A plain ``shutdown()`` is correct
        and fast — ``survivor_detect``'s failed ``wait()`` already aborted the in-flight
        comm, so the destructor's abort is a no-op (does not drain a stuck collective).
        """
        from torchft.process_group import ProcessGroupNCCL

        start = time.monotonic()
        for g in [self._pg, self._gloo_pg]:
            if g is not None:
                g.shutdown()
        self._gloo_pg = None

        self._pg = ProcessGroupNCCL(timeout=timedelta(seconds=self._timeout_s))
        self._pg.configure(
            store_addr=self._store_addr_q1,
            replica_id=str(self._rank),
            rank=0,
            world_size=1,
            quorum_id=1,
        )
        return {"rank": self._rank, "teardown_elapsed_s": round(time.monotonic() - start, 2)}

    def verify_singleton(self) -> dict:
        """Confirm the survivor's new singleton (world_size=1) PG is functional."""
        import torch
        import torch.distributed as dist

        tensor = torch.ones(8, device=self._device) * 7.0
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        start = time.monotonic()
        ok = self._pg.allreduce([tensor], opts).wait()
        return {
            "rank": self._rank,
            "ok": bool(ok),
            "value": tensor[0].item(),
            "elapsed_s": round(time.monotonic() - start, 2),
        }


def _run_reference(
    *,
    timeout_s: float,
    tensor_size: int,
    die_after_s: float,
    teardown_budget_s: float,
) -> None:
    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    base = f"localhost:{store.port}/reference"
    store_addr_q0, store_addr_q1 = f"{base}/q0", f"{base}/q1"

    workers = [_PGWorker.remote() for _ in range(2)]
    init = ray.get(
        [
            w.build_group.remote(
                store_addr_q0=store_addr_q0,
                store_addr_q1=store_addr_q1,
                rank=i,
                world_size=2,
                timeout_s=timeout_s,
            )
            for i, w in enumerate(workers)
        ]
    )
    print(f"built: {init}")

    # 1) Correct collective: SUM allreduce across 2 ranks (values 1.0 + 2.0 == 3.0).
    checks = ray.get([w.allreduce_check.remote(expected=3.0) for w in workers])
    print(f"allreduce: {checks}")
    assert all(c["ok"] for c in checks), checks

    # 2) Dead-peer recovery: victim dies mid-allreduce; survivor detects + reconfigures.
    victim, survivor = workers[0], workers[1]
    victim_ref = victim.run_allreduce_then_die.remote(tensor_size=tensor_size, die_after_s=die_after_s)

    detect = ray.get(survivor.survivor_detect.remote(tensor_size=tensor_size), timeout=die_after_s + timeout_s + 60)
    print(f"detect: {detect}")
    assert detect["status"] in ("wait_false", "exception"), f"peer death not detected: {detect}"
    try:
        ray.get(victim_ref, timeout=die_after_s + 10)
    except Exception as e:
        print(f"victim dead: {type(e).__name__}")

    # ray.get raises GetTimeoutError if reconfigure hangs → the script fails (the point).
    teardown = ray.get(survivor.reconfigure_to_singleton.remote(), timeout=teardown_budget_s)
    print(f"reconfigure: {teardown}")

    verify = ray.get(survivor.verify_singleton.remote(), timeout=30)
    print(f"verify: {verify}")
    assert verify["ok"] and abs(verify["value"] - 7.0) < 1e-6, f"singleton PG not functional: {verify}"

    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print("\nSUCCESS: torchft PG usage + dead-peer reconfigure work correctly (no teardown hang).")


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout (also the userspace abort deadline)")] = 20.0,
    tensor_size: Annotated[
        int, typer.Option(help="allreduce size; larger keeps the collective in-flight longer")
    ] = 100_000_000,
    die_after_s: Annotated[float, typer.Option(help="peer os._exit's this long after starting")] = 2.0,
    teardown_budget_s: Annotated[float, typer.Option(help="ray.get timeout for reconfigure; exceeding = hang")] = 90.0,
) -> None:
    """Run the torchft correct-usage reference (asserts; exits 0 on success)."""
    ray.init(ignore_reinit_error=True)
    _run_reference(
        timeout_s=timeout_s,
        tensor_size=tensor_size,
        die_after_s=die_after_s,
        teardown_budget_s=teardown_budget_s,
    )


if __name__ == "__main__":
    typer.run(main)

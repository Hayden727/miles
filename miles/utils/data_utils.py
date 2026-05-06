from typing import Any

from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions


def compute_dynamic_global_batch_size(args, *, dp_size: int, num_samples: int) -> int:
    """Compute the dynamic global batch size for the given dp_size and sample count.

    HACK: this is a thin wrapper around RolloutManager._compute_dynamic_global_batch_size,
    invoked via a SimpleNamespace mock-self so the trim math + logging stays in one place.
    The wrapper exists because the same logic is needed at training time under delay_split
    FT (where the post-healing dp_size is only known on the training side), but the method
    is currently bound to RolloutManager. Will be cleaned up once that method is refactored
    into a stateless free function.
    """
    from types import SimpleNamespace

    from miles.ray.rollout import RolloutManager

    mock_self = SimpleNamespace(
        train_parallel_config={"dp_size": dp_size},
        args=args,
    )
    return RolloutManager._compute_dynamic_global_batch_size(mock_self, num_samples)


def split_train_data_by_dp(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    rollout_data = {}

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    ans = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_log_probs",
            "rollout_routed_experts",
            "prompt",
            "teacher_log_probs",
            "seq_witness_ids",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        ans.append(rollout_data)
    return ans

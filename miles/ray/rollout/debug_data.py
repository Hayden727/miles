import logging
from pathlib import Path

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def load_debug_rollout_data(args, rollout_id: int) -> tuple[list[Sample], dict]:
    payload = torch.load(
        args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )
    data = [Sample.from_dict(sample) for sample in payload["samples"]]
    # Files recorded before metadata recording have no "metadata" key.
    metadata = payload.get("metadata") or {}
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data, metadata


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool, metadata: dict | None = None) -> None:
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, metadata=metadata or {}, **dump_data), path)

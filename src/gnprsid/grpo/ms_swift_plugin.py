from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from swift.rewards import ORM, orms
except ImportError:  # pragma: no cover - makes local unit tests independent from ms-swift install
    class ORM:  # type: ignore[override]
        pass

    orms = {}

from gnprsid.grpo.reward_current_top10 import compute_score as _default_compute_score


MS_SWIFT_REWARD_PATH_ENV = "GNPRSID_GRPO_REWARD_PATH"
MS_SWIFT_REWARD_NAME_ENV = "GNPRSID_GRPO_REWARD_NAME"


def _row_value(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


@lru_cache(maxsize=1)
def _load_reward_callable():
    reward_path = os.environ.get(MS_SWIFT_REWARD_PATH_ENV)
    reward_name = os.environ.get(MS_SWIFT_REWARD_NAME_ENV, "compute_score")
    if not reward_path:
        return _default_compute_score

    module_path = Path(reward_path)
    spec = importlib.util.spec_from_file_location("gnprsid_grpo_reward_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_callable = getattr(module, reward_name, None)
    if reward_callable is None:
        raise AttributeError(f"Reward function '{reward_name}' not found in {module_path}")
    return reward_callable


class GNPRSIDTop10Reward(ORM):
    def __call__(self, completions, **kwargs):
        reward_callable = _load_reward_callable()
        scores: list[float] = []
        for index, completion in enumerate(completions):
            ground_truth = str(_row_value(kwargs.get("ground_truth"), index))
            extra_info = {
                "sample_id": _row_value(kwargs.get("sample_id"), index),
                "uid": _row_value(kwargs.get("uid"), index),
                "repr": _row_value(kwargs.get("repr"), index),
                "history_source": _row_value(kwargs.get("history_source"), index),
                "target": ground_truth,
                "target_time": _row_value(kwargs.get("target_time"), index),
                "prompt_template_version": _row_value(kwargs.get("prompt_template_version"), index),
            }
            score = reward_callable(
                str(_row_value(kwargs.get("data_source"), index) or ""),
                str(completion),
                ground_truth,
                extra_info=extra_info,
            )
            scores.append(float(score))
        return scores


orms["gnprsid_top10"] = GNPRSIDTop10Reward

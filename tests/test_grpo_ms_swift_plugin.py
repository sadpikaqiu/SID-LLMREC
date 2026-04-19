from gnprsid.grpo.ms_swift_plugin import (
    GNPRSIDTop10Reward,
    MS_SWIFT_REWARD_NAME_ENV,
    MS_SWIFT_REWARD_PATH_ENV,
    _load_reward_callable,
)
from gnprsid.grpo.reward_current_top10 import compute_score


def test_ms_swift_reward_plugin_matches_project_reward(monkeypatch):
    monkeypatch.delenv(MS_SWIFT_REWARD_PATH_ENV, raising=False)
    monkeypatch.delenv(MS_SWIFT_REWARD_NAME_ENV, raising=False)
    _load_reward_callable.cache_clear()

    rewarder = GNPRSIDTop10Reward()
    completion = "<a_1><b_2><c_3> <a_4><b_5><c_6>"
    scores = rewarder(
        [completion],
        data_source=["gnprsid_nyc_sid_current"],
        ground_truth=["<a_1><b_2><c_3>"],
        sample_id=["sample-1"],
        uid=[1],
        repr=["sid"],
        history_source=["current"],
        target_time=["2024-01-01 12:00:00"],
        prompt_template_version=["v3"],
    )

    expected = compute_score(
        "gnprsid_nyc_sid_current",
        completion,
        "<a_1><b_2><c_3>",
        extra_info={
            "sample_id": "sample-1",
            "uid": 1,
            "repr": "sid",
            "history_source": "current",
            "target": "<a_1><b_2><c_3>",
            "target_time": "2024-01-01 12:00:00",
            "prompt_template_version": "v3",
        },
    )
    assert scores == [expected]


def test_ms_swift_reward_plugin_can_load_external_reward_file(monkeypatch, tmp_path):
    reward_file = tmp_path / "reward.py"
    reward_file.write_text(
        "def custom_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    return 1.23 if solution_str == ground_truth else -0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(MS_SWIFT_REWARD_PATH_ENV, str(reward_file))
    monkeypatch.setenv(MS_SWIFT_REWARD_NAME_ENV, "custom_score")
    _load_reward_callable.cache_clear()

    rewarder = GNPRSIDTop10Reward()
    scores = rewarder(
        ["answer", "other"],
        data_source=["x", "x"],
        ground_truth=["answer", "answer"],
    )

    assert scores == [1.23, -0.5]

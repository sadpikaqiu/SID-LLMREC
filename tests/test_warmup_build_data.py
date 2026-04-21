from pathlib import Path
from types import SimpleNamespace

from gnprsid.common.io import iter_jsonl, write_json, write_jsonl
from gnprsid.warmup.build_data import build_ranked_sid_targets, build_warmup_data


def test_build_ranked_sid_targets_prefers_shared_prefixes():
    sid_space = [
        "<a_1><b_1><c_1>",
        "<a_1><b_1><c_1><d_0>",
        "<a_1><b_1><c_2>",
        "<a_1><b_2><c_1>",
        "<a_2><b_1><c_1>",
        "<a_3><b_1><c_1>",
        "<a_4><b_1><c_1>",
        "<a_5><b_1><c_1>",
        "<a_6><b_1><c_1>",
        "<a_7><b_1><c_1>",
    ]
    prefix_groups = {
        "a": {},
        "ab": {},
        "abc": {},
    }
    from gnprsid.warmup.build_data import _build_prefix_groups
    from collections import Counter

    prefix_groups = _build_prefix_groups(sid_space)
    counts = Counter(
        {
            "<a_1><b_1><c_1>": 10,
            "<a_1><b_1><c_1><d_0>": 8,
            "<a_1><b_1><c_2>": 6,
            "<a_1><b_2><c_1>": 4,
        }
    )
    ranked = build_ranked_sid_targets(
        "<a_1><b_1><c_1>",
        sid_space=sid_space,
        target_counts=counts,
        prefix_groups=prefix_groups,
        top_k=10,
    )

    assert ranked[0] == "<a_1><b_1><c_1>"
    assert ranked[1] == "<a_1><b_1><c_1><d_0>"
    assert ranked[2] == "<a_1><b_1><c_2>"
    assert ranked[3] == "<a_1><b_2><c_1>"
    assert len(ranked) == 10


def test_build_ranked_sid_targets_caps_same_abc_prefix_when_alternatives_exist():
    sid_space = [
        "<a_1><b_1><c_1>",
        "<a_1><b_1><c_1><d_0>",
        "<a_1><b_1><c_1><d_1>",
        "<a_1><b_1><c_1><d_2>",
        "<a_1><b_1><c_2>",
        "<a_1><b_2><c_1>",
        "<a_2><b_1><c_1>",
        "<a_3><b_1><c_1>",
        "<a_4><b_1><c_1>",
        "<a_5><b_1><c_1>",
    ]
    from collections import Counter

    from gnprsid.alignment.semantic import sid_prefix
    from gnprsid.warmup.build_data import _build_prefix_groups

    ranked = build_ranked_sid_targets(
        "<a_1><b_1><c_1>",
        sid_space=sid_space,
        target_counts=Counter({sid: len(sid_space) - idx for idx, sid in enumerate(sid_space)}),
        prefix_groups=_build_prefix_groups(sid_space),
        top_k=8,
    )

    target_abc = sid_prefix("<a_1><b_1><c_1>", "abc")
    same_abc_count = sum(1 for sid in ranked if sid_prefix(sid, "abc") == target_abc)
    assert same_abc_count == 2
    assert "<a_1><b_1><c_2>" in ranked
    assert "<a_1><b_2><c_1>" in ranked


def test_build_warmup_data_writes_direct10_targets(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    sid_dir = artifacts / "sid"
    processed.mkdir()
    sid_dir.mkdir(parents=True)

    train_rows = [
        {
            "sample_id": "train-00000-u1",
            "uid": 1,
            "repr": "sid",
            "split": "train",
            "target": "<a_1><b_1><c_1>",
            "target_time": "2024-01-01 12:00:00",
            "key_text": "User_1 visited: <a_1><b_1><c_1> at 2024-01-01 10:00:00",
            "input_text": "User_1 visited: <a_1><b_1><c_1> at 2024-01-01 10:00:00. When 2024-01-01 12:00:00 user_1 is likely to visit:",
        },
        {
            "sample_id": "train-00001-u2",
            "uid": 2,
            "repr": "sid",
            "split": "train",
            "target": "<a_1><b_1><c_1><d_0>",
            "target_time": "2024-01-02 12:00:00",
            "key_text": "User_2 visited: <a_1><b_1><c_1><d_0> at 2024-01-02 10:00:00",
            "input_text": "User_2 visited: <a_1><b_1><c_1><d_0> at 2024-01-02 10:00:00. When 2024-01-02 12:00:00 user_2 is likely to visit:",
        },
    ]
    valid_rows = [dict(train_rows[0], sample_id="val-00000-u1", split="val")]
    write_jsonl(processed / "samples_sid_train.jsonl", train_rows)
    write_jsonl(processed / "samples_sid_val.jsonl", valid_rows)

    sid_payload = {
        str(index): {"sid_token": sid}
        for index, sid in enumerate(
            [
                "<a_1><b_1><c_1>",
                "<a_1><b_1><c_1><d_0>",
                "<a_1><b_1><c_2>",
                "<a_1><b_2><c_1>",
                "<a_2><b_1><c_1>",
                "<a_3><b_1><c_1>",
                "<a_4><b_1><c_1>",
                "<a_5><b_1><c_1>",
                "<a_6><b_1><c_1>",
                "<a_7><b_1><c_1>",
            ],
            start=1,
        )
    }
    write_json(sid_dir / "pid_to_sid.json", sid_payload)

    fake_paths = SimpleNamespace(processed=processed, artifacts=artifacts)
    monkeypatch.setattr("gnprsid.warmup.build_data.dataset_paths", lambda dataset: fake_paths)

    result = build_warmup_data("NYC")
    train_rows_built = list(iter_jsonl(result["train_path"]))

    first = train_rows_built[0]
    outputs = first["output"].split(" ")
    assert first["target"] == outputs[0]
    assert len(outputs) == 10
    assert "exactly 10 complete semantic IDs" in first["instruction"]
    assert "<a_1><b_2><c_3>" in first["instruction"]
    assert "You must return exactly 10 complete semantic IDs" in first["input"]
    assert "<a_1><b_2><c_3>" in first["input"]
    assert result["prompt_template_version"] == "v3"
    assert result["sid_space_size"] == 10

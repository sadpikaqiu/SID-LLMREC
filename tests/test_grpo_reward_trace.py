import json

from gnprsid.grpo.plot_rewards import build_reward_trace_report
from gnprsid.grpo.reward_current_top10 import compute_score


def test_reward_trace_logging_writes_jsonl(monkeypatch, tmp_path):
    trace_dir = tmp_path / "reward-traces"
    monkeypatch.setenv("GNPRSID_REWARD_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("GNPRSID_REWARD_TRACE_GROUP_SIZE", "4")

    score = compute_score(
        "any",
        "<a_1><b_2><c_3>\n<a_4><b_5><c_6>",
        "<a_1><b_2><c_3>",
        extra_info={
            "sample_id": "sample-1",
            "uid": 123,
            "target": "<a_1><b_2><c_3>",
        },
    )

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    rows = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["group_size_hint"] == 4
    assert rows[0]["total_reward"] == score
    assert rows[0]["soft_hit_reward"] == 1.0
    assert rows[0]["format_reward"] > 0.0


def test_build_reward_trace_report_groups_rows_into_synthetic_steps(tmp_path):
    trace_dir = tmp_path / "reward-traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "reward_trace_pid1.jsonl"
    rows = [
        {
            "time_ns": 1,
            "pid": 1,
            "local_record_index": 0,
            "group_size_hint": 2,
            "format_reward": 0.1,
            "reciprocal_rank_reward": 0.2,
            "soft_hit_reward": 1.0,
            "prefix_match_reward": 0.3,
            "diversity_reward": 0.4,
            "total_reward": 2.0,
            "single_line_score": 1.0,
            "valid_count_score": 0.5,
            "exact_ten_score": 0.0,
        },
        {
            "time_ns": 2,
            "pid": 1,
            "local_record_index": 1,
            "group_size_hint": 2,
            "format_reward": 0.3,
            "reciprocal_rank_reward": 0.4,
            "soft_hit_reward": 1.0,
            "prefix_match_reward": 0.5,
            "diversity_reward": 0.6,
            "total_reward": 2.8,
            "single_line_score": 1.0,
            "valid_count_score": 0.7,
            "exact_ten_score": 0.0,
        },
        {
            "time_ns": 3,
            "pid": 1,
            "local_record_index": 2,
            "group_size_hint": 2,
            "format_reward": 0.6,
            "reciprocal_rank_reward": 0.7,
            "soft_hit_reward": 1.0,
            "prefix_match_reward": 0.8,
            "diversity_reward": 0.9,
            "total_reward": 4.0,
            "single_line_score": 1.0,
            "valid_count_score": 1.0,
            "exact_ten_score": 1.0,
        },
        {
            "time_ns": 4,
            "pid": 1,
            "local_record_index": 3,
            "group_size_hint": 2,
            "format_reward": 0.5,
            "reciprocal_rank_reward": 0.6,
            "soft_hit_reward": 1.0,
            "prefix_match_reward": 0.7,
            "diversity_reward": 0.8,
            "total_reward": 3.6,
            "single_line_score": 1.0,
            "valid_count_score": 0.9,
            "exact_ten_score": 1.0,
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output_path = tmp_path / "report.html"
    summary = build_reward_trace_report(trace_dir, output_path=output_path)

    assert summary["trace_row_count"] == 4
    assert summary["synthetic_step_count"] == 2
    assert summary["group_size"] == 2
    assert output_path.exists()
    csv_rows = (tmp_path / "report.csv").read_text(encoding="utf-8")
    assert "step_mean_total_reward" in csv_rows
    assert "cumulative_mean_total_reward" in csv_rows
    html_report = output_path.read_text(encoding="utf-8")
    assert "Synthetic Step" in html_report
    assert "Mean Reward" in html_report
    assert "Cumulative Total Reward" in html_report

import json
import sys

from gnprsid import cli
from gnprsid.grpo.inspect_single_line import inspect_single_line_failures
from gnprsid.grpo.plot_rewards import _downsample_series_map, _downsample_xy, build_reward_trace_report
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
    assert rows[0]["single_line_reward"] == 0.0
    assert rows[0]["valid_count_reward"] > 0.0
    assert rows[0]["exact_ten_reward"] == 0.0
    assert rows[0]["solution_preview"] == "<a_1><b_2><c_3> <a_4><b_5><c_6>"
    assert rows[0]["scored_solution_preview"] == "<a_1><b_2><c_3> <a_4><b_5><c_6>"
    assert rows[0]["parsed_predictions"] == ["<a_1><b_2><c_3>", "<a_4><b_5><c_6>"]


def test_reward_trace_logging_keeps_raw_preview_but_scores_after_leading_think(monkeypatch, tmp_path):
    trace_dir = tmp_path / "reward-traces"
    monkeypatch.setenv("GNPRSID_REWARD_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("GNPRSID_REWARD_TRACE_GROUP_SIZE", "4")

    score = compute_score(
        "any",
        "<think>\nreasoning...\n</think>\n<a_1><b_2><c_3> <a_4><b_5><c_6>",
        "<a_1><b_2><c_3>",
    )

    trace_files = list(trace_dir.glob("*.jsonl"))
    rows = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["solution_preview"].startswith("<think>")
    assert rows[0]["scored_solution_preview"] == "<a_1><b_2><c_3> <a_4><b_5><c_6>"
    assert rows[0]["single_line_score"] == 1.0
    assert rows[0]["total_reward"] == score


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
            "single_line_reward": 0.1,
            "valid_count_reward": 0.125,
            "exact_ten_reward": 0.0,
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
            "single_line_reward": 0.1,
            "valid_count_reward": 0.175,
            "exact_ten_reward": 0.0,
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
            "single_line_reward": 0.1,
            "valid_count_reward": 0.25,
            "exact_ten_reward": 0.15,
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
            "single_line_reward": 0.1,
            "valid_count_reward": 0.225,
            "exact_ten_reward": 0.15,
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
    assert "step_mean_single_line_reward" in csv_rows
    assert "step_mean_exact_ten_reward" in csv_rows
    html_report = output_path.read_text(encoding="utf-8")
    assert "Synthetic Step" in html_report
    assert "Mean Reward" in html_report
    assert "Cumulative Total Reward" in html_report
    assert "Per-Step Mean Reward Components (Split Panels)" in html_report
    assert "Format Reward Components (Weighted)" in html_report
    assert "format_reward" in html_report
    assert "diversity_reward" in html_report
    assert "single_line_reward" in html_report
    assert "exact_ten_reward" in html_report


def test_build_reward_trace_report_defaults_to_outputs_tree(monkeypatch, tmp_path):
    trace_dir = tmp_path / "checkpoints" / "NYC" / "grpo" / "qwen3_8b_sid_current" / "reward_traces"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "reward_trace_pid1.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "time_ns": 1,
                "pid": 1,
                "local_record_index": 0,
                "group_size_hint": 1,
                "format_reward": 0.1,
                "reciprocal_rank_reward": 0.2,
                "soft_hit_reward": 0.3,
                "prefix_match_reward": 0.4,
                "diversity_reward": 0.1,
                "total_reward": 1.1,
                "single_line_score": 1.0,
                "valid_count_score": 1.0,
                "exact_ten_score": 1.0,
                "single_line_reward": 0.1,
                "valid_count_reward": 0.25,
                "exact_ten_reward": 0.15,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakePaths:
        outputs = tmp_path / "outputs" / "NYC"

    monkeypatch.setattr("gnprsid.grpo.plot_rewards.dataset_paths", lambda dataset: FakePaths())

    summary = build_reward_trace_report(trace_dir)

    expected_output = tmp_path / "outputs" / "NYC" / "reports" / "grpo" / "qwen3_8b_sid_current" / "reward_trace_report.html"
    assert summary["output_path"] == str(expected_output)
    assert expected_output.exists()
    assert expected_output.with_suffix(".csv").exists()
    assert expected_output.with_suffix(".summary.json").exists()


def test_downsample_xy_caps_point_count_and_averages_buckets():
    x_values = list(range(1, 11))
    y_values = [float(value) for value in range(10)]

    down_x, down_y = _downsample_xy(x_values, y_values, max_points=4)

    assert len(down_x) <= 4
    assert len(down_x) == len(down_y)
    assert down_x == [3, 6, 9, 10]
    assert down_y == [1.0, 4.0, 7.0, 9.0]


def test_downsample_series_map_uses_shared_buckets():
    x_values = list(range(1, 11))
    series_map = {
        "a": [float(value) for value in range(10)],
        "b": [float(value * 10) for value in range(10)],
    }

    down_x, down_map = _downsample_series_map(x_values, series_map, max_points=4)

    assert down_x == [3, 6, 9, 10]
    assert down_map["a"] == [1.0, 4.0, 7.0, 9.0]
    assert down_map["b"] == [10.0, 40.0, 70.0, 90.0]


def test_build_reward_trace_report_backfills_weighted_format_components(tmp_path):
    trace_dir = tmp_path / "reward-traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "reward_trace_pid1.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "time_ns": 1,
                "pid": 1,
                "local_record_index": 0,
                "group_size_hint": 1,
                "format_reward": 0.45,
                "reciprocal_rank_reward": 0.0,
                "soft_hit_reward": 0.0,
                "prefix_match_reward": 0.0,
                "diversity_reward": 0.0,
                "total_reward": 0.45,
                "single_line_score": 1.0,
                "valid_count_score": 1.0,
                "exact_ten_score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "report.html"
    build_reward_trace_report(trace_dir, output_path=output_path)
    csv_rows = (tmp_path / "report.csv").read_text(encoding="utf-8")

    assert "0.1" in csv_rows
    assert "0.25" in csv_rows
    assert "0.15" in csv_rows


def test_summarize_reward_traces_reports_common_output_patterns(tmp_path):
    from gnprsid.grpo.inspect_trace import summarize_reward_traces

    trace_dir = tmp_path / "reward-traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "reward_trace_pid1.jsonl"
    rows = [
        {
            "solution_preview": "<eos>",
            "parsed_prediction_count": 0,
            "single_line_score": 1.0,
            "total_reward": 0.2,
        },
        {
            "solution_preview": "<eos>",
            "parsed_prediction_count": 0,
            "single_line_score": 1.0,
            "total_reward": 0.2,
        },
        {
            "solution_preview": "<a_1><b_2><c_3>",
            "parsed_prediction_count": 1,
            "single_line_score": 1.0,
            "total_reward": 1.8,
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_reward_traces(trace_dir, top_k=2)

    assert summary["trace_row_count"] == 3
    assert summary["zero_prediction_rate"] == 2 / 3
    assert summary["parsed_prediction_count_histogram"] == {0: 2, 1: 1}
    assert summary["top_solution_previews"][0]["solution_preview"] == "<eos>"
    assert summary["top_solution_previews"][0]["count"] == 2


def test_inspect_single_line_failures_separates_single_and_multi_line_rows(tmp_path):
    trace_dir = tmp_path / "reward-traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "reward_trace_pid1.jsonl"
    rows = [
        {
            "solution_preview": "<a_1><b_2><c_3>",
            "single_line_score": 1.0,
            "parsed_prediction_count": 1,
            "valid_count_score": 0.1,
            "exact_ten_score": 0.0,
            "hit": 1.0,
            "total_reward": 1.2,
        },
        {
            "solution_preview": "<a_1><b_2><c_3> <a_4><b_5><c_6>",
            "single_line_score": 0.0,
            "parsed_prediction_count": 2,
            "valid_count_score": 0.2,
            "exact_ten_score": 0.0,
            "hit": 0.0,
            "total_reward": 0.5,
        },
        {
            "solution_preview": "<a_1><b_2><c_3> <a_7><b_8><c_9>",
            "single_line_score": 0.0,
            "parsed_prediction_count": 2,
            "valid_count_score": 0.2,
            "exact_ten_score": 0.0,
            "hit": 0.0,
            "total_reward": 0.4,
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = inspect_single_line_failures(trace_dir, top_k=2)

    assert summary["single_line_rate"] == 1 / 3
    assert summary["multi_line_rate"] == 2 / 3
    assert summary["single_line_summary"]["row_count"] == 1
    assert summary["multi_line_summary"]["row_count"] == 2
    assert summary["multi_line_summary"]["parsed_prediction_count_histogram"] == {2: 2}
    assert summary["top_multi_line_previews"][0]["count"] == 1


def test_cli_plot_trace_dispatches_to_html_report_builder(monkeypatch):
    calls = {}

    def fake_build_reward_trace_report(**kwargs):
        calls.update(kwargs)
        return {"output_path": "report.html"}

    monkeypatch.setattr("gnprsid.grpo.plot_rewards.build_reward_trace_report", fake_build_reward_trace_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gnprsid.cli",
            "grpo",
            "plot-trace",
            "--trace-path",
            "reward_traces",
            "--output-path",
            "report.html",
            "--csv-path",
            "report.csv",
            "--summary-path",
            "report.summary.json",
            "--group-size",
            "8",
        ],
    )

    cli.main()

    assert calls == {
        "trace_path": "reward_traces",
        "output_path": "report.html",
        "csv_path": "report.csv",
        "summary_path": "report.summary.json",
        "group_size": 8,
    }


def test_cli_inspect_single_line_dispatches(monkeypatch):
    calls = {}

    def fake_inspect_single_line_failures(**kwargs):
        calls.update(kwargs)
        return {"trace_path": "reward_traces"}

    monkeypatch.setattr("gnprsid.grpo.inspect_single_line.inspect_single_line_failures", fake_inspect_single_line_failures)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gnprsid.cli",
            "grpo",
            "inspect-single-line",
            "--trace-path",
            "reward_traces",
            "--top-k",
            "7",
        ],
    )

    cli.main()

    assert calls == {
        "trace_path": "reward_traces",
        "top_k": 7,
    }

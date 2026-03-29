from gnprsid.alignment.evaluate import (
    _evaluate_profile_to_prefix,
    _evaluate_sid_to_abc_profile,
    format_alignment_prompt,
)


def test_format_alignment_prompt_uses_instruction_input_response_blocks():
    prompt = format_alignment_prompt(
        {
            "instruction": "Given a semantic prefix, output its semantic profile JSON.",
            "input": 'Semantic prefix (level abc): <a_1><b_2><c_3>',
        }
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt


def test_evaluate_sid_to_abc_profile_reports_field_metrics():
    records = [
        {
            "sample_id": "s1",
            "task_type": "full_sid_to_abc_profile",
            "source_sid": "<a_1><b_2><c_3><d_0>",
            "source_abc": "<a_1><b_2><c_3>",
            "output": '{"category":"Bar","region":54,"geo_bucket":"G3_5"}',
        },
        {
            "sample_id": "s2",
            "task_type": "full_sid_to_abc_profile",
            "source_sid": "<a_4><b_5><c_6>",
            "source_abc": "<a_4><b_5><c_6>",
            "output": '{"category":"Office","region":24,"geo_bucket":"G1_2"}',
        },
    ]
    predictions = [
        '{"category":"Bar","region":54,"geo_bucket":"G3_5"}',
        '{"category":"Office","region":67,"geo_bucket":"G1_2"}',
    ]
    metrics, samples = _evaluate_sid_to_abc_profile(records, predictions)
    assert metrics["num_samples"] == 2
    assert metrics["valid_profile_rate"] == 1.0
    assert metrics["category_match_rate"] == 1.0
    assert metrics["region_match_rate"] == 0.5
    assert metrics["geo_bucket_match_rate"] == 1.0
    assert metrics["joint_profile_match_rate"] == 0.5
    assert len(samples) == 2


def test_evaluate_profile_to_prefix_checks_candidate_membership():
    records = [
        {
            "sample_id": "s1",
            "task_type": "category_profile_to_a",
            "source_sid": "<a_1>",
            "source_abc": "<a_1><b_2><c_3>",
            "output": "<a_1>",
            "candidate_prefixes": ["<a_1>", "<a_9>", "<a_7>", "<a_8>"],
        }
    ]
    predictions = ["<a_9>"]
    metrics, samples = _evaluate_profile_to_prefix("abc_profile_to_a", records, predictions)
    assert metrics["num_samples"] == 1
    assert metrics["valid_prefix_rate"] == 1.0
    assert metrics["exact_match_rate"] == 0.0
    assert samples[0]["parsed_prefix"] == "<a_9>"

from gnprsid.prompts.render import (
    V2_NEXT_POI_INSTRUCTION,
    build_prompt,
    build_supervised_prompt,
    extract_predictions,
    system_prompt,
)


def sample(repr_name="id"):
    return {
        "sample_id": "test-00001-u1",
        "uid": 1,
        "repr": repr_name,
        "key_text": "User_1 visited: <1> at 2024-01-01 10:00:00, <2> at 2024-01-01 11:00:00",
        "target": "<3>" if repr_name == "id" else "<a_1><b_2><c_3>",
        "target_time": "2024-01-01 12:00:00",
    }


def test_retrieval_prompt_has_explicit_sections():
    prompt = build_prompt(
        sample("id"),
        "retrieval",
        similar_map={"test-00001-u1": [{"sample_id": "train-00001-u1", "score": 0.9}]},
        bank_map={
            "train-00001-u1": {
                "key_text": "User_1 visited: <4> at 2024-01-01 09:00:00",
                "target": "<5>",
                "target_time": "2024-01-01 10:00:00",
            }
        },
        top_k_retrieval=1,
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert V2_NEXT_POI_INSTRUCTION in prompt
    assert "Retrieved similar cases:" in prompt
    assert "Case 1:" in prompt
    assert "Current trajectory to predict:" not in prompt


def test_supervised_prompt_uses_single_target_requirement():
    prompt = build_supervised_prompt(sample("id"), "current")
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "Output format:" not in prompt


def test_system_prompt_candidate_count_is_configurable():
    single = system_prompt("sid", "current", candidate_count=1)
    topk = system_prompt("sid", "current", candidate_count=10)
    assert "helpful assistant" in single
    assert "semantic IDs only" in single
    assert "exactly 10 complete semantic IDs" in topk
    assert "<a_1><b_2><c_3>" in topk
    assert "single space" in topk
    assert single != topk


def test_direct_semantic_prompt_omits_example_tail():
    prompt = build_prompt(sample("sid"), "current", candidate_count=10)
    assert "You must return exactly 10 complete semantic IDs" in prompt
    assert "<a_1><b_2><c_3>" in prompt
    assert "Start the reply immediately with the first semantic ID." in prompt
    assert "Example:" not in prompt
    assert "Semantic ID notes:" not in prompt


def test_extract_predictions_requires_complete_sid_from_a_prefix():
    parsed = extract_predictions("><b_10><c_9> <a_9><b_10><c_9><d_0>", "sid")
    assert parsed == ["<a_9><b_10><c_9><d_0>"]

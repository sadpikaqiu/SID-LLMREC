from gnprsid.prompts.render import build_prompt, build_supervised_prompt


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
    assert "Retrieved similar cases:" in prompt
    assert "Observed trajectory:" in prompt
    assert "Ground-truth next POI" in prompt
    assert "Current trajectory to predict:" in prompt


def test_supervised_prompt_uses_single_target_requirement():
    prompt = build_supervised_prompt(sample("id"), "current")
    assert "exactly 1" in prompt
    assert "exactly 10" not in prompt

from gnprsid.alignment.evaluate import (
    _extract_sid,
    _shared_prefix_length,
    format_alignment_prompt,
)


def test_format_alignment_prompt_uses_instruction_input_response_blocks():
    prompt = format_alignment_prompt(
        {
            "instruction": "Given POI attributes, describe its semantic ID.",
            "input": "Attributes: { Category: Cafe; Region: 1 }",
        }
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt


def test_extract_sid_returns_first_sid():
    text = "The answer is <a_1><b_2><c_3> and not something else."
    assert _extract_sid(text) == "<a_1><b_2><c_3>"


def test_shared_prefix_length_handles_partial_match():
    target = "<a_1><b_2><c_3><d_0>"
    assert _shared_prefix_length("<a_1><b_2><c_8>", target) == 2
    assert _shared_prefix_length("<a_9><b_2><c_3>", target) == 0

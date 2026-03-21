from __future__ import annotations


RESPONSE_TEMPLATE = "### Response:\n"


def format_instruction_completion(batch) -> list[str]:
    instructions = batch.get("instruction", [])
    inputs = batch.get("input", [""] * len(instructions))
    outputs = batch.get("output", [""] * len(instructions))

    texts: list[str] = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        texts.append(
            "### Instruction:\n"
            f"{str(instruction).strip()}\n\n"
            "### Input:\n"
            f"{str(input_text).strip()}\n\n"
            "### Response:\n"
            f"{str(output_text).strip()}<|eot_id|>"
        )
    return texts

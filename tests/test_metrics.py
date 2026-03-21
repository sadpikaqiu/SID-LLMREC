from gnprsid.eval.metrics import accuracy_at_k, evaluate_prediction_records


def test_accuracy_at_k():
    assert accuracy_at_k(["<1>", "<2>", "<3>"], "<2>", 2) is True
    assert accuracy_at_k(["<1>", "<2>", "<3>"], "<3>", 2) is False


def test_evaluate_prediction_records_deduplicates_from_parsed_predictions():
    metrics, records = evaluate_prediction_records(
        [
            {
                "repr": "id",
                "target": "<2>",
                "prediction": "<2> <2> <3>",
                "parsed_predictions": ["<2>", "<3>"],
                "prompt": "x",
                "prompt_char_length": 1,
            }
        ]
    )
    assert metrics["acc_at_1"] == 1.0
    assert records[0]["top1_correct"] is True

from math import isclose

from gnprsid.grpo.reward_current_single_sid import compute_score


def test_grpo_reward_full_credit_for_valid_exact_output():
    target = "<a_1><b_2><c_3>"
    prediction = "<a_1><b_2><c_3>"
    expected = 0.1 + 1.0 + 0.2
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_keeps_prefix_credit_when_format_is_invalid():
    target = "<a_1><b_2><c_3>"
    prediction = "<a_1><b_2><c_3> explanation"
    expected = 0.2
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_multiple_predictions_void_format_and_exact():
    target = "<a_1><b_2><c_3>"
    prediction = "<a_1><b_2><c_3> <a_4><b_5><c_6>"
    expected = 0.2
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_only_counts_a_b_c_prefix_even_when_d_differs():
    target = "<a_1><b_2><c_3><d_0>"
    prediction = "<a_1><b_2><c_3><d_9>"
    expected = 0.1 + 0.2
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_ignores_truncated_suffixes_that_do_not_start_with_a():
    target = "<a_1><b_2><c_3>"
    prediction = "><b_2><c_3> <a_4><b_5><c_6>"
    expected = 0.025
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)

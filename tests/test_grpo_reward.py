from math import isclose

from gnprsid.grpo.reward_current_top10 import compute_score


def test_grpo_reward_full_credit_for_valid_ranked_output():
    target = "<a_1><b_2><c_3>"
    prediction = (
        "<a_1><b_2><c_3> <a_4><b_5><c_6> <a_7><b_8><c_9> <a_10><b_11><c_12> "
        "<a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> <a_22><b_23><c_24> "
        "<a_25><b_26><c_27> <a_28><b_29><c_30>"
    )
    expected = 0.5 + 1.9 + 1.0 + 0.2 * (3 / 30) + 0.1
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_keeps_hit_and_rank_when_multiline_format_is_invalid():
    target = "<a_1><b_2><c_3>"
    prediction = "<a_1><b_2><c_3>\n<a_4><b_5><c_6>"
    expected = 0.5 * (0.5 * (2 / 10)) + 1.9 + 1.0 + 0.2 * (3 / 30) + 0.1 * (2 / 10)
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_duplicate_predictions_reduce_diversity_and_void_format():
    target = "<a_1><b_2><c_3>"
    prediction = (
        "<a_1><b_2><c_3> <a_1><b_2><c_3> <a_4><b_5><c_6> <a_7><b_8><c_9> "
        "<a_10><b_11><c_12> <a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> "
        "<a_22><b_23><c_24> <a_25><b_26><c_27>"
    )
    expected = 0.5 * (0.2 + 0.5 * (9 / 10)) + 1.9 + 1.0 + 0.2 * (3 / 30) + 0.1 * (9 / 10)
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_only_counts_a_b_c_prefix_even_when_d_differs():
    target = "<a_1><b_2><c_3><d_0>"
    prediction = (
        "<a_1><b_2><c_3><d_9> <a_4><b_5><c_6> <a_7><b_8><c_9> <a_10><b_11><c_12> "
        "<a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> <a_22><b_23><c_24> "
        "<a_25><b_26><c_27> <a_28><b_29><c_30>"
    )
    expected = 0.5 + 0.2 * (3 / 30) + 0.1
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_ignores_truncated_suffixes_that_do_not_start_with_a():
    target = "<a_1><b_2><c_3>"
    prediction = "><b_2><c_3> <a_4><b_5><c_6>"
    expected = 0.5 * (0.2 + 0.5 * (1 / 10)) + 0.1 * (1 / 10)
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_boosts_top_ranks_more_than_plain_inverse_rank():
    target = "<a_4><b_5><c_6>"
    prediction = (
        "<a_1><b_2><c_3> <a_4><b_5><c_6> <a_7><b_8><c_9> <a_10><b_11><c_12> "
        "<a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> <a_22><b_23><c_24> "
        "<a_25><b_26><c_27> <a_28><b_29><c_30>"
    )
    expected = 0.5 + (1.0 / 2.0) * 1.8 + 1.0 + 0.2 * (3 / 30) + 0.1
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)


def test_grpo_reward_ignores_leading_think_block_when_scoring_format():
    target = "<a_1><b_2><c_3>"
    prediction = (
        "<think>\nreasoning...\n</think>\n"
        "<a_1><b_2><c_3> <a_4><b_5><c_6> <a_7><b_8><c_9> <a_10><b_11><c_12> "
        "<a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> <a_22><b_23><c_24> "
        "<a_25><b_26><c_27> <a_28><b_29><c_30>"
    )
    expected = 0.5 + 1.9 + 1.0 + 0.2 * (3 / 30) + 0.1
    assert isclose(compute_score("any", prediction, target), expected, abs_tol=1e-7)

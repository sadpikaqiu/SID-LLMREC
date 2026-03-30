from gnprsid.alignment.semantic import (
    choose_hard_negative_prefixes,
    compute_geo_bucket,
    forward_profile_sampling_weight,
    profile_for_level,
    profile_to_json,
    sid_level,
    sid_prefix,
)


def test_compute_geo_bucket_clips_upper_bounds():
    assert compute_geo_bucket(10.0, 20.0, 0.0, 10.0, 0.0, 20.0, grid_size=8) == "G7_7"


def test_sid_prefix_extracts_expected_levels():
    token = "<a_1><b_2><c_3><d_4>"
    assert sid_prefix(token, "a") == "<a_1>"
    assert sid_prefix(token, "ab") == "<a_1><b_2>"
    assert sid_prefix(token, "abc") == "<a_1><b_2><c_3>"
    assert sid_level(token) == "full_sid"


def test_profile_to_json_uses_stable_field_order():
    profile = profile_for_level("abc", category="Bar", region=54, geo_bucket="G3_5")
    assert profile_to_json(profile, level="abc") == '{"category":"Bar","region":54,"geo_bucket":"G3_5"}'


def test_choose_hard_negative_prefixes_prefers_region_and_geo_conflicts_for_abc():
    prefix_profiles = {
        "<a_1><b_2><c_3>": {
            "profile": {"category": "Bar", "region": 54, "geo_bucket": "G3_5"},
        },
        "<a_1><b_2><c_4>": {
            "profile": {"category": "Bar", "region": 67, "geo_bucket": "G3_5"},
        },
        "<a_1><b_2><c_5>": {
            "profile": {"category": "Bar", "region": 54, "geo_bucket": "G3_4"},
        },
        "<a_9><b_9><c_9>": {
            "profile": {"category": "Office", "region": 54, "geo_bucket": "G3_4"},
        },
    }
    negatives = choose_hard_negative_prefixes(
        level="abc",
        positive_prefix="<a_1><b_2><c_3>",
        positive_profile=prefix_profiles["<a_1><b_2><c_3>"]["profile"],
        prefix_profiles=prefix_profiles,
        negative_count=3,
        rng=__import__("random").Random(42),
    )
    assert "<a_1><b_2><c_4>" in negatives
    assert "<a_1><b_2><c_5>" in negatives


def test_forward_profile_sampling_weight_is_mild_not_extreme():
    common = forward_profile_sampling_weight(
        "Bar",
        54,
        "G3_3",
        __import__("collections").Counter({"Bar": 100}),
        __import__("collections").Counter({54: 200}),
        __import__("collections").Counter({"G3_3": 180}),
    )
    rare = forward_profile_sampling_weight(
        "Aquarium",
        53,
        "G0_4",
        __import__("collections").Counter({"Aquarium": 5}),
        __import__("collections").Counter({53: 8}),
        __import__("collections").Counter({"G0_4": 6}),
    )
    assert 0.1 <= common <= 1.0
    assert rare > common

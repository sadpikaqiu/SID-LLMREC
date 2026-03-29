from gnprsid.alignment.semantic import (
    compute_geo_bucket,
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

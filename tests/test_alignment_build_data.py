from gnprsid.alignment.build_data import (
    _build_phase_b_records,
    _mixed_phase_train_records,
    _split_abc_prefixes,
)


def test_split_abc_prefixes_keeps_train_and_valid_disjoint():
    train_abc, valid_abc = _split_abc_prefixes(
        ["<a_1><b_1><c_1>", "<a_1><b_1><c_2>", "<a_1><b_1><c_3>", "<a_1><b_1><c_4>"],
        valid_ratio=0.25,
        seed=42,
    )
    assert train_abc.isdisjoint(valid_abc)
    assert len(train_abc | valid_abc) == 4


def test_phase_b_records_do_not_create_profile_to_full_sid_task():
    prefix_profiles = {
        "a": {
            "<a_1>": {"profile": {"category": "Bar"}, "profile_json": '{"category":"Bar"}'},
        },
        "ab": {
            "<a_1><b_2>": {
                "profile": {"category": "Bar", "region": 54},
                "profile_json": '{"category":"Bar","region":54}',
            },
        },
        "abc": {
            "<a_1><b_2><c_3>": {
                "profile": {"category": "Bar", "region": 54, "geo_bucket": "G3_5"},
                "profile_json": '{"category":"Bar","region":54,"geo_bucket":"G3_5"}',
            },
        },
    }
    semantic_rows = [
        {
            "full_sid": "<a_1><b_2><c_3><d_0>",
            "abc": "<a_1><b_2><c_3>",
        }
    ]
    abc_records, reverse_records, full_sid_records = _build_phase_b_records(
        ["<a_1><b_2><c_3>"],
        semantic_rows,
        prefix_profiles,
        __import__("random").Random(42),
    )
    task_types = {record["task_type"] for record in [*abc_records, *reverse_records, *full_sid_records]}
    assert "profile_to_full_sid" not in task_types
    assert "full_sid_to_abc_profile" in task_types


def test_mixed_phase_train_records_uses_requested_task_mix():
    full_record = {"task_type": "full_sid_to_abc_profile", "category": "Bar", "region": 54, "geo_bucket": "G3_3"}
    abc_record = {"task_type": "abc_to_abc_profile", "category": "Bar", "region": 54, "geo_bucket": "G3_3"}
    reverse_record = {"task_type": "abc_profile_to_abc", "category": "Bar", "region": 54, "geo_bucket": "G3_3"}
    replay_record = {"task_type": "a_to_category_profile", "category": "Bar", "region": None, "geo_bucket": None}
    records = {
        "full_sid_to_abc_profile": [dict(full_record) for _ in range(10)],
        "abc_to_abc_profile": [dict(abc_record) for _ in range(10)],
        "abc_profile_to_abc": [dict(reverse_record) for _ in range(10)],
        "phase_a_replay": [dict(replay_record) for _ in range(10)],
    }
    mixed = _mixed_phase_train_records(
        families=records,
        task_mix={
            "full_sid_to_abc_profile": 0.50,
            "abc_to_abc_profile": 0.20,
            "abc_profile_to_abc": 0.20,
            "phase_a_replay": 0.10,
        },
        category_counts=__import__("collections").Counter({"Bar": 10}),
        region_counts=__import__("collections").Counter({54: 10}),
        geo_bucket_counts=__import__("collections").Counter({"G3_3": 10}),
        seed=42,
    )
    assert len(mixed) == 40
    assert {item["task_type"] for item in mixed} == {
        "full_sid_to_abc_profile",
        "abc_to_abc_profile",
        "abc_profile_to_abc",
        "a_to_category_profile",
    }

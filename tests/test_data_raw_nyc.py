from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from gnprsid.data.raw_nyc import build_nyc_from_raw


def _write_synthetic_raw(path: Path) -> None:
    rows = []
    base = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)

    user_specs = [
        (100, "poi_a", "cat_a", "Cafe", 40.7128, -74.0060),
        (200, "poi_b", "cat_b", "Park", 40.7580, -73.9855),
    ]

    for step in range(30):
        for uid, pid, category_id, category, lat, lon in user_specs:
            ts = base + timedelta(hours=(step * 2) + (0 if uid == 100 else 1))
            rows.append(
                "\t".join(
                    [
                        str(uid),
                        pid,
                        category_id,
                        category,
                        f"{lat:.4f}",
                        f"{lon:.4f}",
                        "0",
                        ts.strftime("%a %b %d %H:%M:%S %z %Y"),
                    ]
                )
            )

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_nyc_from_raw_generates_current_project_shapes(tmp_path: Path) -> None:
    raw_path = tmp_path / "NYC.txt"
    dataset_root = tmp_path / "NYC"
    _write_synthetic_raw(raw_path)

    manifest = build_nyc_from_raw(dataset="NYC", raw_path=raw_path, output_root=dataset_root)

    processed = dataset_root / "processed"
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")
    test_df = pd.read_csv(processed / "test.csv")
    history_df = pd.read_csv(processed / "history.csv")
    poi_info_df = pd.read_csv(processed / "poi_info.csv")
    pid_mapping_df = pd.read_csv(processed / "pid_mapping.csv")
    uid_mapping_df = pd.read_csv(processed / "uid_mapping.csv")
    cat_mapping_df = pd.read_csv(processed / "catname_mapping.csv")
    region_mapping_df = pd.read_csv(processed / "region_mapping.csv")

    assert manifest["counts"]["raw_rows"] == 60
    assert manifest["counts"]["filtered_rows"] == 60
    assert len(train_df) == 2
    assert len(val_df) == 2
    assert len(test_df) == 2
    assert len(history_df) == 2
    assert len(poi_info_df) == 2
    assert len(pid_mapping_df) == 2
    assert len(uid_mapping_df) == 2
    assert len(cat_mapping_df) == 2
    assert len(region_mapping_df) == 2

    assert pid_mapping_df.to_dict("records") == [
        {"Original_Pid": "poi_a", "Mapped_Pid": 1},
        {"Original_Pid": "poi_b", "Mapped_Pid": 2},
    ]
    assert uid_mapping_df.to_dict("records") == [
        {"Original_Uid": 100, "Mapped_Uid": 1},
        {"Original_Uid": 200, "Mapped_Uid": 2},
    ]

    assert train_df["Uid"].tolist() == [1, 2]
    assert val_df["Uid"].tolist() == [1, 2]
    assert test_df["Uid"].tolist() == [1, 2]
    assert history_df["Uid"].tolist() == [1, 2]

    assert poi_info_df.columns.tolist() == [
        "Pid",
        "Uid",
        "Catname",
        "Region",
        "Time",
        "neighbors",
        "forward_neighbors",
        "original_pid",
        "category",
        "category_id",
        "latitude",
        "longitude",
        "unique_user_count",
        "visit_count",
        "visit_hours",
        "visit_time_and_count",
        "pid",
        "region",
    ]

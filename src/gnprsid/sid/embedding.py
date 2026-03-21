from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from gnprsid.common.io import ensure_dir


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find any of columns {candidates} in dataframe.")


def column2vec(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    n_components: int | None = None,
    column_candidates: list[str] | None = None,
    output_name: str = "column_to_embedding.pkl",
) -> Path:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    output_dir = ensure_dir(output_dir)
    df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")
    column_name = _resolve_column(df, column_candidates or ["category"])
    values = [str(value) for value in df[column_name].dropna().unique().tolist()]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(values, show_progress_bar=True)
    if n_components is not None and n_components < embeddings.shape[1]:
        embeddings = PCA(n_components=n_components).fit_transform(embeddings)
    value_to_embedding = dict(zip(values, embeddings))
    output_path = output_dir / output_name
    with output_path.open("wb") as handle:
        pickle.dump(value_to_embedding, handle)
    return output_path


def category2vec(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    n_components: int | None = None,
    category_column: str | None = None,
) -> Path:
    return column2vec(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name=model_name,
        n_components=n_components,
        column_candidates=[category_column] if category_column else ["category", "Category", "Catname", "catname"],
        output_name="category_to_embedding.pkl",
    )


def region2vec(
    csv_path: str | Path,
    output_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    n_components: int | None = None,
    region_column: str | None = None,
) -> Path:
    return column2vec(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name=model_name,
        n_components=n_components,
        column_candidates=[region_column] if region_column else ["region", "Region"],
        output_name="region_to_embedding.pkl",
    )


def parse_time_like(value) -> list[int] | dict:
    if pd.isna(value) or value == "" or value == "{}" or value == "[]":
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed


def extract_time_features(value) -> np.ndarray:
    parsed = parse_time_like(value)
    if isinstance(parsed, dict):
        if not parsed:
            return np.zeros(12)
        hours = np.array([int(hour) for hour in parsed.keys()])
        counts = np.array(list(parsed.values()), dtype=float)
    elif isinstance(parsed, list):
        if not parsed:
            return np.zeros(12)
        hours = np.array([int(hour) for hour in parsed], dtype=int)
        counts = np.ones(len(hours), dtype=float)
    else:
        return np.zeros(12)

    total = counts.sum()
    if total == 0:
        return np.zeros(12)
    weights = counts / total
    fourier_features = []
    for k in range(1, 7):
        sin_val = np.sum(weights * np.sin(2 * np.pi * k * hours / 24.0))
        cos_val = np.sum(weights * np.cos(2 * np.pi * k * hours / 24.0))
        fourier_features.extend([sin_val, cos_val])
    return np.array(fourier_features)


def latlon_to_3d(lat, lon) -> np.ndarray:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def build_poi_embedding_dict(
    poi_info_csv: str | Path,
    category_embedding_pkl: str | Path,
    output_dir: str | Path,
    region_embedding_pkl: str | Path | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    df = pd.read_csv(poi_info_csv, on_bad_lines="skip", encoding="utf-8")
    with Path(category_embedding_pkl).open("rb") as handle:
        category_to_embedding: Dict[str, np.ndarray] = pickle.load(handle)

    pid_col = _resolve_column(df, ["pid", "Pid"])
    category_col = _resolve_column(df, ["category", "Category", "Catname", "catname"])
    time_col = _resolve_column(df, ["visit_time_and_count", "Time", "time"])

    has_latlon = {"latitude", "longitude"}.issubset(df.columns) or {"Latitude", "Longitude"}.issubset(df.columns)
    region_to_embedding: Dict[str, np.ndarray] | None = None
    if has_latlon:
        lat_col = _resolve_column(df, ["latitude", "Latitude", "lat", "Lat"])
        lon_col = _resolve_column(df, ["longitude", "Longitude", "lon", "Lon"])
    else:
        if region_embedding_pkl is None:
            raise ValueError(
                "Region embedding file is required when poi_info.csv does not provide latitude/longitude columns."
            )
        with Path(region_embedding_pkl).open("rb") as handle:
            region_to_embedding = pickle.load(handle)
        region_col = _resolve_column(df, ["region", "Region"])

    cat_dim = len(next(iter(category_to_embedding.values())))
    region_dim = len(next(iter(region_to_embedding.values()))) if region_to_embedding else 0
    vectors = {}
    for _, row in df.iterrows():
        cat_vec = category_to_embedding.get(str(row[category_col]), np.zeros(cat_dim))
        if has_latlon:
            spatial_vec = latlon_to_3d(float(row[lat_col]), float(row[lon_col]))
        else:
            spatial_vec = region_to_embedding.get(str(row[region_col]), np.zeros(region_dim))
        time_vec = extract_time_features(row[time_col])
        vectors[int(row[pid_col])] = np.concatenate([cat_vec, spatial_vec, time_vec])

    output_path = output_dir / "poi_emb_dict.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(vectors, handle)
    return output_path

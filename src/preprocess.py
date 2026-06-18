"""
Sommel-AI :: Data preprocessing
-------------------------------
Loads the Wine Enthusiast 130k dataset, cleans text fields, imputes missing
prices by (variety, country) median (then country median, then global median),
and writes a cleaned parquet file used by the recommender and classifier.

Run once:
    python src/preprocess.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default input -- accepts either uploads/ path or a local data/ path.
DEFAULT_INPUTS = [
    Path("/sessions/nifty-nice-bohr/mnt/uploads/winemag-data-130k-v2.csv"),
    ROOT / "data" / "winemag-data-130k-v2.csv",
]

OUTPUT = DATA_DIR / "wines_clean.pkl"


# -----------------------------------------------------------------------------
# Text cleaning
# -----------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NONALPHA = re.compile(r"[^a-zA-Z\s\-']")


def clean_text(s: str) -> str:
    """Lowercase, strip non-letters, collapse whitespace. Cheap & deterministic."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _NONALPHA.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def extract_vintage(title: str) -> float:
    """Pull a 4-digit year out of the wine title; NaN if not found."""
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"\b(19|20)\d{2}\b", title)
    return float(m.group()) if m else np.nan


# -----------------------------------------------------------------------------
# Price imputation
# -----------------------------------------------------------------------------
def impute_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy (most-specific to most-general):
      1. median price by (variety, country)
      2. median price by country
      3. global median
    Tracks an `price_imputed` boolean column so downstream code can flag rows.
    """
    df = df.copy()
    df["price_imputed"] = df["price"].isna()

    by_var_country = df.groupby(["variety", "country"])["price"].transform("median")
    df["price"] = df["price"].fillna(by_var_country)

    by_country = df.groupby("country")["price"].transform("median")
    df["price"] = df["price"].fillna(by_country)

    df["price"] = df["price"].fillna(df["price"].median())
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def find_input() -> Path:
    for p in DEFAULT_INPUTS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find winemag-data-130k-v2.csv. Place it at "
        f"{DEFAULT_INPUTS[1]} or pass --in <path>."
    )


def main() -> int:
    inp = find_input()
    print(f"[preprocess] loading {inp}", flush=True)
    df = pd.read_csv(inp, index_col=0)
    print(f"[preprocess] loaded {len(df):,} rows, {df.shape[1]} cols", flush=True)

    # Force plain numpy / Python-object dtypes throughout. Newer pandas (3.x)
    # on Python 3.14 defaults string columns to PyArrow-backed ExtensionArrays,
    # which silently break numpy boolean masking and sklearn indexing
    # downstream. Normalize once here so every consumer gets predictable types.
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col].dtype):
            df[col] = df[col].astype(object)

    # Drop rows with no description -- the only field we absolutely need.
    df = df.dropna(subset=["description", "variety"]).reset_index(drop=True)

    # Cleaned description for TF-IDF.
    df["description_clean"] = df["description"].map(clean_text)

    # Pull vintage out of the title for display.
    df["vintage"] = df["title"].map(extract_vintage)

    # Impute price.
    df = impute_price(df)

    # Value score primitives: rating-per-dollar, log-scaled so a $5 wine
    # doesn't dominate the rankings.
    df["value_raw"] = df["points"] / np.log1p(df["price"])

    # Normalize value within variety so rankings are fair across categories
    # (a "good value Pinot" vs a "good value Bordeaux").
    grp = df.groupby("variety")["value_raw"]
    df["value_score"] = (df["value_raw"] - grp.transform("mean")) / grp.transform("std").replace(0, 1)
    df["value_score"] = df["value_score"].fillna(0).clip(-3, 3)

    # Persist (pickle is version-stable across pandas releases; parquet needs
    # matching pyarrow versions which causes friction across machines)
    print(f"[preprocess] writing {OUTPUT}", flush=True)
    df.to_pickle(OUTPUT)
    print(f"[preprocess] done: {len(df):,} rows, {df.shape[1]} cols", flush=True)

    # Quick sanity print
    print("\n[preprocess] summary:")
    print(f"  unique varieties : {df['variety'].nunique():,}")
    print(f"  unique countries : {df['country'].nunique():,}")
    print(f"  price range      : ${df['price'].min():.0f}–${df['price'].max():.0f}")
    print(f"  points range     : {df['points'].min()}–{df['points'].max()}")
    print(f"  imputed prices   : {df['price_imputed'].sum():,} ({df['price_imputed'].mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Sommel-AI :: Model training
---------------------------
Builds and persists the artifacts the live app needs:

  1. TF-IDF vectorizer fit on cleaned wine descriptions
  2. Sparse TF-IDF matrix of all wines (used for cosine-similarity lookup)
  3. KNN classifier predicting wine variety from description vectors
     -- limited to the top-N most common varieties for tractable accuracy

Run once after preprocess.py:
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wines_clean.pkl"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

VEC_PATH = ART / "tfidf_vectorizer.joblib"
MAT_PATH = ART / "tfidf_matrix.joblib"
META_PATH = ART / "wines_meta.pkl"
KNN_PATH = ART / "knn_classifier.joblib"
KNN_LABELS_PATH = ART / "knn_labels.joblib"
METRICS_PATH = ART / "metrics.txt"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Sized to fit comfortably in Streamlit Community Cloud's 1 GB RAM ceiling.
# Locally you can bump max_features back to 20000, ngram_range=(1,2), and
# TOP_N_VARIETIES to 25 for slightly higher classifier accuracy (~3-5 points).
TFIDF_KW = dict(
    max_features=5000,
    ngram_range=(1, 1),     # unigrams only -- halves vocab space
    min_df=5,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True,
)

# Classifier is trained only on the top-N varieties -- the long tail of 707
# varieties has too few samples to learn cleanly.
TOP_N_VARIETIES = 12
KNN_K = 15
SKIP_EVAL = True            # skip held-out metrics on cloud to save peak RAM


def main() -> int:
    if not DATA.exists():
        print(f"[train] missing {DATA}; run preprocess.py first")
        return 1

    print(f"[train] loading {DATA}", flush=True)
    df = pd.read_pickle(DATA)
    print(f"[train] {len(df):,} rows", flush=True)

    # Defensive dtype normalization (newer pandas may wrap strings in PyArrow
    # ExtensionArrays which silently break sklearn / numpy indexing).
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col].dtype):
            df[col] = df[col].astype(object)

    # --- TF-IDF over all wines ---
    print("[train] fitting TF-IDF on description_clean", flush=True)
    vec = TfidfVectorizer(**TFIDF_KW)
    # .tolist() forces Python strings -- avoids any ExtensionArray weirdness.
    X = vec.fit_transform(df["description_clean"].fillna("").astype(str).tolist())
    print(f"[train] TF-IDF matrix: {X.shape}, nnz={X.nnz:,}", flush=True)

    joblib.dump(vec, VEC_PATH)
    joblib.dump(X, MAT_PATH)

    # Lightweight metadata used at query time (small enough to mmap).
    meta_cols = [
        "title", "variety", "country", "province", "region_1", "winery",
        "points", "price", "vintage", "value_score", "price_imputed",
        "description",
    ]
    df[meta_cols].to_pickle(META_PATH)
    print(f"[train] saved vectorizer + matrix + metadata", flush=True)

    # --- KNN classifier on top-N varieties ---
    print(f"[train] training KNN on top-{TOP_N_VARIETIES} varieties", flush=True)
    top_varieties = df["variety"].value_counts().head(TOP_N_VARIETIES).index
    # Force numpy/object dtypes: newer pandas (3.x / Streamlit Cloud Python 3.14)
    # defaults to PyArrow-backed string arrays which sklearn cannot index with
    # numpy integer arrays. .to_numpy().astype(str) gives us a plain ndarray.
    mask = df["variety"].isin(top_varieties).to_numpy()
    Xc = X[mask]
    yc = df.loc[mask, "variety"].to_numpy().astype(str)
    print(f"[train] classifier data: {Xc.shape[0]:,} rows, {len(top_varieties)} classes", flush=True)

    if SKIP_EVAL:
        # Train on the full classifier dataset, skip held-out evaluation
        # (held-out predict on KNN over 10k+ sparse vectors is the peak-RAM
        # operation in this script). Metrics from a richer local run are
        # already in artifacts/metrics.txt for the proposal.
        knn = KNeighborsClassifier(
            n_neighbors=KNN_K, metric="cosine", algorithm="brute", n_jobs=1
        )
        knn.fit(Xc, yc)
        metrics_text = (
            f"# Sommel-AI KNN classifier (no held-out eval)\n"
            f"top_n_varieties: {TOP_N_VARIETIES}\n"
            f"k:               {KNN_K}\n"
            f"train rows:      {Xc.shape[0]:,}\n"
        )
        METRICS_PATH.write_text(metrics_text)
        print("[train] trained KNN; eval skipped", flush=True)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xc, yc, test_size=0.15, random_state=42, stratify=yc
        )
        knn = KNeighborsClassifier(
            n_neighbors=KNN_K, metric="cosine", algorithm="brute", n_jobs=-1
        )
        knn.fit(X_tr, y_tr)
        print("[train] evaluating on held-out test set", flush=True)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(X_te.shape[0], size=min(3000, X_te.shape[0]), replace=False)
        y_pred = knn.predict(X_te[sample_idx])
        y_true = y_te[sample_idx]
        report = classification_report(y_true, y_pred, zero_division=0)
        accuracy = (y_pred == y_true).mean()
        metrics = [
            "# Sommel-AI KNN classifier metrics",
            f"top_n_varieties: {TOP_N_VARIETIES}",
            f"k:               {KNN_K}",
            f"train rows:      {X_tr.shape[0]:,}",
            f"test rows:       {X_te.shape[0]:,} (eval sample {len(sample_idx):,})",
            f"accuracy:        {accuracy:.4f}",
            "",
            report,
        ]
        METRICS_PATH.write_text("\n".join(metrics))
        print(f"[train] accuracy on sample = {accuracy:.4f}", flush=True)
        print(f"[train] full report -> {METRICS_PATH}", flush=True)

    joblib.dump(knn, KNN_PATH)
    joblib.dump(list(top_varieties), KNN_LABELS_PATH)
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

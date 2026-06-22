"""
Sommel-AI :: Model training
---------------------------
Builds and persists the artifacts the live app needs:

  1. TF-IDF vectorizer fit on cleaned wine descriptions
  2. Sparse TF-IDF matrix of all wines (used for cosine-similarity lookup)
  3. Logistic Regression classifier predicting wine variety from description
     vectors -- limited to the top-N most common varieties.

We swapped the original KNN classifier for Logistic Regression because:
  - 40x smaller artifact (~0.5 MB vs ~21 MB for KNN)
  - ~5-8 percentage point higher accuracy on TF-IDF text features
  - Sub-millisecond prediction at query time (12 dot products vs 77k cosine
    comparisons)
  - Native probability outputs (predict_proba) for confidence UI

KNN is still trained when SKIP_EVAL=False and reported alongside LogReg in the
metrics file -- this gives the proposal's methodology section a clean baseline
vs. deployed-model comparison.

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
from sklearn.linear_model import LogisticRegression
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

# Deployed classifier (was KNN, now Logistic Regression).
# Path name kept stable so the recommender code doesn't need to change.
CLASSIFIER_PATH = ART / "classifier.joblib"
CLASSIFIER_LABELS_PATH = ART / "classifier_labels.joblib"
METRICS_PATH = ART / "metrics.txt"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Sized to fit comfortably in Streamlit Community Cloud's 1 GB RAM ceiling.
# Locally you can bump max_features back to 20000, ngram_range=(1,2), and
# TOP_N_VARIETIES to 25 for slightly higher classifier accuracy.
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

# Varieties to always include in the classifier even if they're outside the
# top-N. Used to make sure the prediction banner can return a sensible
# sparkling-category variety when the user picks color=sparkling. (Sparkling
# Blend has 2,153 reviews, Champagne Blend 1,396 -- below top-12 but plenty
# for the classifier to learn.)
MUST_INCLUDE_VARIETIES = [
    "Sparkling Blend",
    "Champagne Blend",
]

# Logistic Regression config -- defaults are fine for sparse TF-IDF text.
LOGREG_KW = dict(
    max_iter=300,
    n_jobs=-1,
    solver="lbfgs",
    C=1.0,           # L2 regularization strength
    random_state=42,
)

# KNN config (kept for the baseline comparison run).
KNN_K = 15

SKIP_EVAL = True     # skip held-out metrics on cloud to save peak RAM


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
    X = vec.fit_transform(df["description_clean"].fillna("").astype(str).tolist())
    print(f"[train] TF-IDF matrix: {X.shape}, nnz={X.nnz:,}", flush=True)

    joblib.dump(vec, VEC_PATH)
    joblib.dump(X, MAT_PATH)

    # Lightweight metadata used at query time.
    meta_cols = [
        "title", "variety", "country", "province", "region_1", "winery",
        "points", "price", "vintage", "value_score", "price_imputed",
        "description",
    ]
    df[meta_cols].to_pickle(META_PATH)
    print(f"[train] saved vectorizer + matrix + metadata", flush=True)

    # --- Build classifier dataset (top-N varieties + must-include) ---
    print(f"[train] preparing classifier data for top-{TOP_N_VARIETIES} + "
          f"must-include varieties", flush=True)
    top_varieties = list(df["variety"].value_counts().head(TOP_N_VARIETIES).index)
    present = set(df["variety"].unique())
    for v in MUST_INCLUDE_VARIETIES:
        if v in present and v not in top_varieties:
            top_varieties.append(v)
    print(f"[train] classifier classes: {top_varieties}", flush=True)
    mask = df["variety"].isin(top_varieties).to_numpy()
    Xc = X[mask]
    yc = df.loc[mask, "variety"].to_numpy().astype(str)
    print(f"[train] classifier data: {Xc.shape[0]:,} rows, {len(top_varieties)} classes", flush=True)

    if SKIP_EVAL:
        # Cloud / fast mode: train LogReg on the full classifier dataset,
        # skip the held-out evaluation step.
        print("[train] training LogisticRegression (no held-out eval)", flush=True)
        clf = LogisticRegression(**LOGREG_KW)
        clf.fit(Xc, yc)
        metrics_text = (
            f"# Sommel-AI deployed classifier: LogisticRegression\n"
            f"top_n_varieties:  {TOP_N_VARIETIES}\n"
            f"train rows:       {Xc.shape[0]:,}\n"
            f"features:         {Xc.shape[1]:,}\n"
            f"solver:           {LOGREG_KW['solver']}\n"
            f"C:                {LOGREG_KW['C']}\n"
            "# (held-out evaluation skipped to fit Streamlit Cloud RAM ceiling;\n"
            "#  run locally with SKIP_EVAL=False for full KNN-vs-LogReg comparison)\n"
        )
        METRICS_PATH.write_text(metrics_text)
        print("[train] trained LogReg; eval skipped", flush=True)
    else:
        # Local mode: train BOTH KNN (baseline) and LogReg (deployed) on the
        # same train/test split, report a side-by-side comparison.
        print("[train] running KNN-vs-LogReg comparison (held-out eval)", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xc, yc, test_size=0.15, random_state=42, stratify=yc
        )

        # --- Baseline: KNN ---
        print("[train]   training KNN baseline", flush=True)
        knn = KNeighborsClassifier(
            n_neighbors=KNN_K, metric="cosine", algorithm="brute", n_jobs=-1
        )
        knn.fit(X_tr, y_tr)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(X_te.shape[0], size=min(3000, X_te.shape[0]), replace=False)
        knn_pred = knn.predict(X_te[sample_idx])
        y_true = y_te[sample_idx]
        knn_acc = (knn_pred == y_true).mean()
        knn_report = classification_report(y_true, knn_pred, zero_division=0)
        print(f"[train]   KNN accuracy = {knn_acc:.4f}", flush=True)

        # --- Deployed: LogReg ---
        print("[train]   training LogisticRegression", flush=True)
        clf = LogisticRegression(**LOGREG_KW)
        clf.fit(X_tr, y_tr)
        clf_pred = clf.predict(X_te)
        clf_acc = (clf_pred == y_te).mean()
        clf_report = classification_report(y_te, clf_pred, zero_division=0)
        print(f"[train]   LogReg accuracy = {clf_acc:.4f}", flush=True)

        # --- Side-by-side report for the proposal ---
        metrics = [
            "# Sommel-AI classifier comparison",
            f"top_n_varieties: {TOP_N_VARIETIES}",
            f"train rows:      {X_tr.shape[0]:,}",
            f"test rows:       {X_te.shape[0]:,}",
            f"features:        {X_tr.shape[1]:,}",
            "",
            f"KNN baseline (k={KNN_K}, cosine)   accuracy: {knn_acc:.4f}",
            f"LogisticRegression  (deployed)    accuracy: {clf_acc:.4f}",
            f"Improvement                                {(clf_acc - knn_acc) * 100:+.2f} pp",
            "",
            "=== KNN baseline report ===",
            knn_report,
            "",
            "=== LogReg deployed report ===",
            clf_report,
        ]
        METRICS_PATH.write_text("\n".join(metrics))
        print(f"[train] comparison report -> {METRICS_PATH}", flush=True)

    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(list(top_varieties), CLASSIFIER_LABELS_PATH)
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Sommel-AI :: Recommendation engine
----------------------------------
Loads persisted TF-IDF artifacts and exposes a single `recommend()` call.

Ranking model
-------------
For each candidate wine i we compute:

    final_score(i) = alpha * sim(query, i) + beta * value_score(i)

Where:
  - sim is cosine similarity between the (expanded) query vector and wine i.
  - value_score is the precomputed within-variety z-score of (points / log(1+price)).
  - alpha + beta = 1, default alpha=0.75 so similarity dominates but value
    breaks ties between similarly-described wines.

Optional filters: max budget (hard cut), preferred food (boosts varieties that
pair with that food via the food_pairing knowledge base).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .food_pairing import food_text, keywords_for, principle_for, varieties_for_food
from .preprocess import clean_text

# -----------------------------------------------------------------------------
# Artifact paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
VEC_PATH = ART / "tfidf_vectorizer.joblib"
MAT_PATH = ART / "tfidf_matrix.joblib"
META_PATH = ART / "wines_meta.pkl"
# Classifier artifact (LogisticRegression, formerly KNN). Falls back to the
# older filename so old deploys still load while the new artifacts build.
CLASSIFIER_PATH = ART / "classifier.joblib"
CLASSIFIER_LABELS_PATH = ART / "classifier_labels.joblib"
KNN_PATH_LEGACY = ART / "knn_classifier.joblib"


# -----------------------------------------------------------------------------
# Lazy-loaded singletons
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _vectorizer():
    return joblib.load(VEC_PATH)


@lru_cache(maxsize=1)
def _matrix():
    return joblib.load(MAT_PATH)


@lru_cache(maxsize=1)
def _meta() -> pd.DataFrame:
    df = pd.read_pickle(META_PATH).reset_index(drop=True)
    # Defensive: re-normalize any ExtensionArray columns to plain numpy/object.
    # The pickle was written by preprocess.py which already does this, but if
    # an older artifact is loaded we still want safe behavior.
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col].dtype):
            df[col] = df[col].astype(object)
    return df


@lru_cache(maxsize=1)
def _classifier():
    """Load the deployed classifier. Prefers the new LogReg artifact, falls
    back to any legacy KNN artifact still on disk."""
    if CLASSIFIER_PATH.exists():
        return joblib.load(CLASSIFIER_PATH)
    if KNN_PATH_LEGACY.exists():
        return joblib.load(KNN_PATH_LEGACY)
    return None


# Backwards-compatibility shim so older code paths that called _knn() still work.
_knn = _classifier


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@dataclass
class Recommendation:
    rank: int
    title: str
    variety: str
    country: str
    province: str
    winery: str
    points: int
    price: float
    vintage: Optional[float]
    similarity: float
    value_score: float
    final_score: float
    description: str
    pairings: List[str]
    explanation: str


def predict_variety(query: str, color: Optional[str] = None) -> Optional[str]:
    """Use the deployed classifier to predict the most likely variety for a query.

    If ``color`` is given (red / white / rosé / sparkling), the prediction is
    constrained to varieties matching that color -- so a user who picks "white"
    will never be told "you'd love a Cabernet Sauvignon" even if their query
    sounds red. Uses predict_proba() to rank all classes and returns the top
    one that's color-eligible.
    """
    clf = _classifier()
    if clf is None:
        return None
    vec = _vectorizer()
    q = vec.transform([clean_text(query)])

    # No color filter: just take argmax.
    if not color:
        return str(clf.predict(q)[0])

    wanted = _varieties_for_color(color)
    if not wanted:
        return str(clf.predict(q)[0])

    # Pull class probabilities and pick the top variety that matches the color.
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(q)[0]
    else:
        # LinearSVC fallback: use decision_function as a pseudo-score.
        probs = clf.decision_function(q)[0]
    classes = list(clf.classes_)
    ranked = sorted(range(len(classes)), key=lambda i: -probs[i])
    for idx in ranked:
        if classes[idx].lower() in wanted:
            return str(classes[idx])
    # No color-eligible class in the classifier's vocabulary: return top
    # overall rather than nothing, so the banner still works.
    return str(classes[ranked[0]])


def recommend(
    query: str,
    food: Optional[str] = None,
    max_price: Optional[float] = None,
    color: Optional[str] = None,
    top_n: int = 8,
    alpha: float = 0.75,
    use_classifier_boost: bool = True,
) -> List[Recommendation]:
    """
    Return ranked wine recommendations for a natural-language query.

    Parameters
    ----------
    query : str
        Free-text description of taste, mood, occasion.
    food : str, optional
        A dish or food keyword (e.g. "steak", "salmon"). Wines whose variety
        traditionally pairs with the food get a similarity boost.
    max_price : float, optional
        Hard upper bound on USD price.
    color : str, optional
        "red", "white", "rosé", "sparkling" -- restricts to varieties of that color.
    top_n : int
        Number of recommendations to return.
    alpha : float
        Blend weight on similarity (vs value score). 0.75 by default.
    use_classifier_boost : bool
        If True, ask the KNN classifier which variety the query "sounds like"
        and add a small bonus to that variety's candidates.
    """
    vec = _vectorizer()
    X = _matrix()
    meta = _meta()

    # --- 1. Build the query vector ---
    q_clean = clean_text(query)

    # Mix in food/flavor keywords for the predicted variety -- this is the
    # "classifier-guided query expansion" trick.
    predicted_variety: Optional[str] = None
    if use_classifier_boost and _classifier() is not None:
        try:
            # Pass color through so the predicted variety stays consistent
            # with the user's color filter (avoids "you'd love a Cabernet"
            # when the user picked white).
            predicted_variety = predict_variety(query, color=color)
            foods_kw, flavors_kw = keywords_for(predicted_variety)
            q_clean = " ".join([q_clean] + foods_kw[:3] + flavors_kw[:3])
        except Exception:
            predicted_variety = None

    # If the user named a food, fold its pairing keywords into the query too.
    if food:
        q_clean = q_clean + " " + clean_text(food)
        food_varieties = set(varieties_for_food(food))
    else:
        food_varieties = set()

    q_vec = vec.transform([q_clean])

    # --- 2. Filter the candidate set ---
    # Each sub-mask is wrapped in np.asarray() so a pandas ExtensionArray
    # never leaks into the boolean accumulator.
    mask = np.ones(len(meta), dtype=bool)
    if max_price is not None:
        mask &= np.asarray(meta["price"].values <= float(max_price), dtype=bool)
    if color:
        wanted = _varieties_for_color(color)
        if wanted:
            mask &= np.asarray(meta["variety"].str.lower().isin(wanted).values, dtype=bool)

    if mask.sum() == 0:
        # Fall back to ignoring the color filter rather than returning nothing.
        if max_price is not None:
            mask = np.asarray(meta["price"].values <= float(max_price), dtype=bool)
        else:
            mask = np.ones(len(meta), dtype=bool)

    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return []

    # --- 3. Cosine similarity over candidates only (sparse-safe) ---
    sims = cosine_similarity(q_vec, X[cand_idx]).ravel()

    # --- 4. Final score = alpha*sim + beta*value, with optional food boost ---
    beta = 1.0 - alpha
    value = meta["value_score"].values[cand_idx]
    # value is z-scored to ~[-3,3]; squash to ~[0,1] for comparable units with sim.
    value_norm = (value + 3.0) / 6.0
    final = alpha * sims + beta * value_norm

    if food_varieties:
        boost = np.array(
            [0.05 if v.lower() in food_varieties else 0.0
             for v in meta["variety"].values[cand_idx]]
        )
        final = final + boost

    if predicted_variety:
        pv = predicted_variety.lower()
        bump = np.array(
            [0.03 if v.lower() == pv else 0.0
             for v in meta["variety"].values[cand_idx]]
        )
        final = final + bump

    # --- 5. Top-N ---
    order = np.argsort(-final)[: top_n * 3]   # over-pick then dedupe by winery
    seen_wineries: set = set()
    picks: List[int] = []
    for j in order:
        winery = meta["winery"].iat[cand_idx[j]]
        if winery in seen_wineries:
            continue
        seen_wineries.add(winery)
        picks.append(j)
        if len(picks) == top_n:
            break

    # --- 6. Materialize Recommendation objects ---
    out: List[Recommendation] = []
    for rank, j in enumerate(picks, start=1):
        i = cand_idx[j]
        row = meta.iloc[i]
        foods_kw, _ = keywords_for(row["variety"])
        out.append(
            Recommendation(
                rank=rank,
                title=row["title"],
                variety=row["variety"],
                country=row["country"] or "",
                province=row["province"] or "",
                winery=row["winery"] or "",
                points=int(row["points"]),
                price=float(row["price"]),
                vintage=row["vintage"] if not pd.isna(row["vintage"]) else None,
                similarity=float(sims[j]),
                value_score=float(value[j]),
                final_score=float(final[j]),
                description=row["description"],
                pairings=foods_kw[:5],
                explanation=_explanation(
                    sims[j], value[j], predicted_variety, row["variety"], food
                ),
            )
        )
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_RED = {
    "pinot noir", "cabernet sauvignon", "merlot", "syrah", "shiraz",
    "zinfandel", "malbec", "nebbiolo", "sangiovese", "tempranillo",
    "red blend", "bordeaux-style red blend", "portuguese red", "grenache",
    "cabernet franc", "petit verdot", "carmenère", "carmenere",
}
_WHITE = {
    "chardonnay", "sauvignon blanc", "riesling", "pinot grigio", "pinot gris",
    "gewürztraminer", "viognier", "chenin blanc", "white blend",
    "albariño", "albarino", "moscato", "semillon",
}
_ROSE = {"rosé", "rose"}
_SPARKLING = {"champagne blend", "sparkling blend", "prosecco"}


def _varieties_for_color(color: str) -> set:
    c = color.lower().strip()
    return {
        "red": _RED,
        "white": _WHITE,
        "rosé": _ROSE,
        "rose": _ROSE,
        "sparkling": _SPARKLING,
    }.get(c, set())


def _explanation(sim: float, value: float, predicted: Optional[str],
                 actual: str, food: Optional[str]) -> str:
    bits = [f"Matches your description (sim={sim:.2f})"]
    if value > 0.5:
        bits.append("strong value for its category")
    elif value < -0.5:
        bits.append("premium / aspirational pick")
    if predicted and predicted.lower() == (actual or "").lower():
        bits.append(f"the classifier thought your query sounded like a {predicted}")
    if food:
        foods_kw, _ = keywords_for(actual)
        if any(food.lower() in f.lower() for f in foods_kw):
            bits.append(f"traditional pairing with {food}")
        # Surface the underlying pairing principle (from the Wine Matchmaker
        # guide) when one applies — gives the user the *why*.
        rule = principle_for(food)
        if rule:
            bits.append(f"rule: {rule}")
    return " — ".join(bits) + "."

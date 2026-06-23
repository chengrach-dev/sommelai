"""
Sommel-AI :: Streamlit demo
---------------------------
Run from the project root:

    streamlit run app.py

The app expects the artifacts in ./artifacts -- produce them with:
    python src/preprocess.py
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when streamlit launches us.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- Auto-bootstrap artifacts on first deploy --------------------------------
# Streamlit Community Cloud (and any fresh checkout) won't have the .pkl /
# .joblib artifacts. Build them once on first launch; subsequent launches reuse
# them from disk. Total cold-start: ~30s.
def _bootstrap_artifacts():
    needed = [
        ROOT / "data" / "wines_clean.pkl",
        ROOT / "artifacts" / "tfidf_matrix.joblib",
        ROOT / "artifacts" / "classifier.joblib",
    ]
    must_rebuild = not all(p.exists() for p in needed)

    # Even if the classifier artifact exists, verify it knows about the
    # must-include varieties (e.g. Sparkling Blend). If not, the artifact is
    # stale from an earlier training run and we need to rebuild.
    if not must_rebuild:
        try:
            import joblib
            from src.train import MUST_INCLUDE_VARIETIES
            classes = list(joblib.load(ROOT / "artifacts" / "classifier.joblib").classes_)
            for v in MUST_INCLUDE_VARIETIES:
                if v not in classes:
                    must_rebuild = True
                    break
        except Exception:
            must_rebuild = True

    # Verify the cleaned data has the inflation-adjusted price columns
    # (price_2017 was added when we introduced the 2026 inflation factor).
    # If the cached wines_clean.pkl predates that change, rebuild it so
    # downstream prices reflect 2026 dollars.
    if not must_rebuild:
        try:
            import pandas as pd
            df_head = pd.read_pickle(ROOT / "data" / "wines_clean.pkl")
            if "price_2017" not in df_head.columns:
                must_rebuild = True
        except Exception:
            must_rebuild = True

    if not must_rebuild:
        return

    # Force-delete stale data so preprocess regenerates it from the CSV.
    stale = ROOT / "data" / "wines_clean.pkl"
    if stale.exists():
        try:
            stale.unlink()
        except Exception:
            pass

    import streamlit as _st
    with _st.spinner("First-time setup: cleaning data and training models (~30s)…"):
        from src.preprocess import main as preprocess_main
        from src.train import main as train_main
        if not (ROOT / "data" / "wines_clean.pkl").exists():
            preprocess_main()
        train_main()

_bootstrap_artifacts()

from src.recommender import recommend, predict_variety   # noqa: E402

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sommel-AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .wine-card {
        background: #FAF7F2;
        border: 1px solid #E5DDD3;
        border-left: 4px solid #722F37;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
      }
      .wine-title { font-size: 1.05rem; font-weight: 600; color: #2D1B1E; }
      .wine-meta  { color: #6B5D52; font-size: 0.9rem; margin-bottom: 6px; }
      .wine-tag   { display:inline-block; background:#722F37; color:white;
                    padding:2px 10px; border-radius:12px; font-size:0.75rem;
                    margin-right:6px; }
      .wine-tag.value { background:#3D6B3A; }
      .wine-tag.score { background:#4A4A6B; }
      .small  { color:#8A7A6D; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
st.sidebar.title("🍷 Sommel-AI")
st.sidebar.caption("Describe what you want. We'll find the wine.")

query = st.sidebar.text_area(
    "Describe the wine you want",
    value="something light, crisp, and citrusy for a hot summer evening",
    height=110,
    help="Plain English — flavor, mood, occasion, vibe. No wine jargon required.",
)

food = st.sidebar.text_input(
    "Food you're pairing with (optional)",
    value="grilled salmon",
    help="A dish, ingredient, or cuisine. We'll find wines that go well with it.",
)

color = st.sidebar.selectbox(
    "Wine color (optional)",
    options=["any", "red", "white", "rosé", "sparkling"],
    index=0,
)

max_price = st.sidebar.slider(
    "Max price (USD)", min_value=5, max_value=200, value=30, step=5,
    help="Prices are 2017 Wine Enthusiast retail prices adjusted for "
         "inflation (×1.33, BLS CPI Dec 2017 → 2026).",
)

with st.sidebar.expander("Advanced"):
    top_n = st.slider("Number of results", 3, 15, 10)
    alpha = st.slider(
        "Similarity vs. value weight",
        0.0, 1.0, 0.75, 0.05,
        help="1.0 = pure description match. 0.0 = pure best-value-for-money.",
    )
    use_classifier = st.checkbox(
        "Use classifier-guided query expansion", value=True,
        help="Ask the KNN classifier what variety your query sounds like, "
             "and use that variety's flavor keywords to enrich the query.",
    )

go = st.sidebar.button("Find me wines", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------
st.title("Sommel-AI")
st.markdown(
    "Tell us what you're in the mood for — a flavor, a dish, a vibe, a budget — "
    "and we'll find wines you'll love. No wine-speak required."
)

if not go:
    st.info(
        "Fill in the sidebar and click **Find me wines** to get started. "
        "Try queries like *“off-dry aromatic white for spicy Thai curry”* or "
        "*“bold structured red for a steak dinner”*."
    )
    st.stop()

if not query.strip():
    st.error("Please describe what you're looking for.")
    st.stop()


# -----------------------------------------------------------------------------
# Run recommender
# -----------------------------------------------------------------------------
with st.spinner("Tasting through 130,000 wines..."):
    predicted = None
    if use_classifier:
        try:
            # Pass color so the banner respects the user's color preference.
            predicted = predict_variety(
                query, color=None if color == "any" else color
            )
        except Exception:
            predicted = None

    results = recommend(
        query=query,
        food=food if food.strip() else None,
        max_price=float(max_price),
        color=None if color == "any" else color,
        top_n=top_n,
        alpha=alpha,
        use_classifier_boost=use_classifier,
    )


# -----------------------------------------------------------------------------
# Top banner: what the classifier thinks
# -----------------------------------------------------------------------------
if predicted:
    st.markdown(
        f"""
        <div style="background:#FAF7F2; border:1px solid #E5DDD3;
                    border-left:4px solid #722F37; border-radius:6px;
                    padding:18px 22px; margin:8px 0 20px 0;">
          <div style="color:#6B5D52; font-size:0.95rem; margin-bottom:6px;">
            Based on what you described, we think you'd love a
          </div>
          <div style="color:#2D1B1E; font-size:1.9rem; font-weight:700;
                      line-height:1.2; word-break:break-word;">
            {predicted} 🍷
          </div>
          <div style="color:#6B5D52; font-size:0.9rem; margin-top:10px;">
            Here are our top picks for you below.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if not results:
    st.warning("No wines matched. Try loosening the price limit or color filter.")
    st.stop()

st.subheader(f"Top {len(results)} recommendations")


def _value_label(z: float) -> str:
    """Convert within-variety value z-score into a plain-English label."""
    if z >= 1.5:  return "Great deal"
    if z >= 0.5:  return "Good value"
    if z >= -0.5: return "Fair price"
    if z >= -1.5: return "Splurge pick"
    return "Premium pick"


def _match_label(sim: float) -> str:
    """Convert cosine similarity into a plain-English match strength."""
    if sim >= 0.40: return "Strong match"
    if sim >= 0.25: return "Good match"
    if sim >= 0.15: return "Decent match"
    return "Worth a try"


def _points_label(pts: int) -> str:
    """Convert Wine Enthusiast 100-pt score into a familiar descriptor."""
    if pts >= 95: return f"Outstanding ({pts}/100)"
    if pts >= 90: return f"Excellent ({pts}/100)"
    if pts >= 85: return f"Very good ({pts}/100)"
    return f"Good ({pts}/100)"


for r in results:
    vintage = f"{int(r.vintage)}" if r.vintage else "—"
    pairing_line = ", ".join(r.pairings) if r.pairings else "—"
    st.markdown(
        f"""
        <div class="wine-card">
          <div class="wine-title">#{r.rank}. {r.title}</div>
          <div class="wine-meta">
            {r.variety} · {r.winery} · {r.province}, {r.country} · vintage {vintage}
          </div>
          <div style="margin: 6px 0;">
            <span class="wine-tag">{_points_label(r.points)}</span>
            <span class="wine-tag">${r.price:.0f}</span>
            <span class="wine-tag value">{_value_label(r.value_score)}</span>
            <span class="wine-tag score">{_match_label(r.similarity)}</span>
          </div>
          <div style="margin-top: 10px;">
            <em>“{r.description}”</em>
          </div>
          <div class="small" style="margin-top: 8px;">
            <b>Pairs well with:</b> {pairing_line}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Footer: methodology summary (for the demo recording)
# -----------------------------------------------------------------------------
with st.expander("How it works (methodology)"):
    st.markdown(
        """
        **Data.** 129,970 cleaned reviews from the Wine Enthusiast 130k
        dataset (Kaggle, scraped 2017). Missing prices (~7%) imputed by
        `(variety, country) → country → global` medians; descriptions
        lowercased and stripped of non-letter tokens.

        **Inflation adjustment.** The original 2017 prices are multiplied by
        a 1.33× CPI-based factor (US Bureau of Labor Statistics CPI-U;
        cumulative inflation Dec 2017 → 2026) so the dollar amounts shown
        reflect roughly current retail. The original column is preserved as
        `price_2017` for auditability.

        **NLP pipeline.** `TfidfVectorizer(max_features=5000, ngram_range=(1,1),
        min_df=5, max_df=0.85, sublinear_tf=True, stop_words="english")` on
        cleaned descriptions, producing a sparse 130k × 5k matrix.

        **Recommender.** For a query, we compute cosine similarity between the
        query vector and all wines that pass the price + color filters. Final
        ranking is

        ```
        final = α · cosine_sim + (1-α) · value_score
              + 0.05 × (variety pairs with food)
              + 0.03 × (variety equals classifier prediction)
        ```

        where `value_score` is the within-variety z-score of
        `points / log(1 + price)`. Top-N picks are de-duplicated by winery so
        a single producer doesn't dominate the list.

        **Classifier.** Logistic Regression (multinomial, lbfgs, L2-regularized)
        trained on TF-IDF vectors of the top 12 most-reviewed varieties
        (~77k rows, ~12k held-out). Held-out test accuracy ≈ **0.79** on a
        12-way classification (random baseline ≈ 0.08). The model's prediction
        is used to enrich the query with the predicted variety's pairing /
        flavor keywords ("classifier-guided query expansion").

        **Baseline comparison.** We baselined with a KNN classifier (k=15,
        cosine metric) on the same data and split. KNN reached 0.68 accuracy;
        LogReg outperformed it by **+11 percentage points** and produces a
        ~45× smaller artifact (~0.5 MB vs ~21 MB), so LogReg is the deployed
        model.

        **Food pairing.** A curated `variety → (foods, flavors)` lookup table
        plus a `cuisine → varieties` index and a `principle → rule of thumb`
        table, drawn from Wine Folly, Court of Master Sommeliers (US &
        Europe), MasterClass, The Wine Matchmaker (Antony Anderson, WSET L3),
        Thermador's Ultimate Wine Pairing Guide, and H-E-B's Wine Pairing
        Chart. Covers 30+ varieties and 11 cuisines.
        """
    )

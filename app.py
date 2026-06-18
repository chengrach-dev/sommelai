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
        ROOT / "artifacts" / "knn_classifier.joblib",
    ]
    if all(p.exists() for p in needed):
        return
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
    help="A dish, ingredient, or cuisine. We boost wines that traditionally pair well with it.",
)

color = st.sidebar.selectbox(
    "Wine color (optional)",
    options=["any", "red", "white", "rosé", "sparkling"],
    index=0,
)

max_price = st.sidebar.slider(
    "Max price (USD)", min_value=5, max_value=200, value=30, step=5
)

with st.sidebar.expander("Advanced"):
    top_n = st.slider("Number of results", 3, 12, 6)
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
    "*Natural-language wine recommendations* — describe what you want like you'd "
    "tell a friend, and a content-based recommender trained on 130,000 wine reviews "
    "will rank candidates by description match and price-quality value."
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
            predicted = predict_variety(query)
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
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Your query sounds like", predicted)
    with c2:
        st.caption(
            "The KNN classifier was trained on 100k+ wine reviews across the top "
            "25 varieties. It mapped your description to the variety it most "
            "resembles — we use that to enrich the search."
        )

st.divider()


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if not results:
    st.warning("No wines matched. Try loosening the price limit or color filter.")
    st.stop()

st.subheader(f"Top {len(results)} recommendations")

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
            <span class="wine-tag">{r.points} pts</span>
            <span class="wine-tag">${r.price:.0f}</span>
            <span class="wine-tag value">value {r.value_score:+.2f}</span>
            <span class="wine-tag score">match {r.similarity:.2f}</span>
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
        **Data.** 129,970 cleaned reviews from the Wine Enthusiast 130k dataset
        (Kaggle). Missing prices imputed by `(variety, country) → country → global`
        medians; descriptions lowercased and stripped of non-letter tokens.

        **NLP pipeline.** `TfidfVectorizer(max_features=20k, ngram=(1,2), min_df=5,
        sublinear_tf=True)` on cleaned descriptions, producing a sparse 130k × 20k
        matrix.

        **Recommender.** For a query, we compute cosine similarity between the
        query vector and all wines that pass the price + color filters. Final
        ranking is

        ```
        final = α · cosine_sim + (1-α) · value_score
              + 0.05 × (variety pairs with food)
              + 0.03 × (variety equals classifier prediction)
        ```

        where `value_score` is the within-variety z-score of `points / log(1 + price)`.

        **Classifier.** KNN (k=15, cosine metric) over the TF-IDF vectors, trained
        on the top 25 varieties (~100k rows). Test-set accuracy ≈ 0.57 on a 25-way
        classification (random baseline ≈ 0.04). The prediction is used to expand
        the query with the predicted variety's pairing/flavor keywords.

        **Food pairing.** A hand-curated `variety → (foods, flavors)` lookup table
        built from standard sommelier references (Wine Folly, Court of Master
        Sommeliers).
        """
    )

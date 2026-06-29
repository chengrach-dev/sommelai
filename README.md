# Sommel-AI

A natural-language wine recommendation engine. Tell it what you want in plain English ("light, citrusy white for grilled salmon under $25", "German Riesling for sushi", "Bordeaux for a steak dinner"), and it returns ranked wine picks from 130,000 Wine Enthusiast reviews — balanced for description match, price-quality value, food pairing, and country/region preference.

Built for Cornell Johnson's BANA 6920 Machine Learning Applications in Business by Team 5.

## What's inside

```
sommelai/
├── app.py                  # Streamlit demo UI with consumer-friendly score labels
├── requirements.txt
├── DEPLOY.md               # Streamlit Community Cloud deployment guide
├── data/
│   ├── winemag-data-130k-v2.csv   # Kaggle Wine Reviews dataset
│   └── wines_clean.pkl            # produced by preprocess.py
├── artifacts/              # produced by train.py
│   ├── tfidf_vectorizer.joblib
│   ├── tfidf_matrix.joblib
│   ├── wines_meta.pkl
│   ├── classifier.joblib          # deployed Logistic Regression
│   ├── classifier_labels.joblib   # variety labels (14 classes)
│   └── metrics.txt                # held-out KNN-vs-LogReg comparison
└── src/
    ├── preprocess.py       # cleaning, price imputation, inflation adjustment, value score
    ├── train.py            # TF-IDF + Logistic Regression (with KNN baseline)
    ├── recommender.py      # ranking engine: cosine sim + value + boosts + origin filter
    └── food_pairing.py     # variety / cuisine / origin / principle knowledge base
```

## Quickstart

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. the Kaggle dataset is already at data/winemag-data-130k-v2.csv

# 3. build artifacts (~45 seconds)
python src/preprocess.py
python src/train.py

# 4. run the demo
streamlit run app.py
```

The Streamlit app auto-bootstraps artifacts on first launch, so you can also skip steps 3 and just run `streamlit run app.py` — it'll build everything on the first request.

## Live deployment

Deployed on Streamlit Community Cloud. See `DEPLOY.md` for the full deployment walkthrough.

## Methodology in one breath

| Layer | What it does |
|---|---|
| **Preprocessing** | Cleans descriptions, imputes missing prices by `(variety, country) → country → global` median, inflation-adjusts 2017 → 2026 dollars using BLS CPI (~1.33×), computes within-variety value z-score. |
| **TF-IDF** | `max_features=5000, ngram_range=(1,1), min_df=5, max_df=0.85, sublinear_tf=True` on cleaned descriptions. Produces a 130k × 5k sparse matrix. |
| **Recommender** | `final = α · cosine_sim + (1-α) · value_score`, plus small boosts when the variety pairs with the user's food and when it matches the classifier's prediction. Hard filters for max price, color, and country/region. |
| **Classifier (deployed)** | Logistic Regression (multinomial, lbfgs, L2-regularized) on top-12 most-reviewed varieties + Sparkling Blend + Champagne Blend (14 classes total, ~80k training rows). Used as a query-expansion signal — predicts the variety the query "sounds like" and adds that variety's flavor/food keywords. Supports `predict_proba()` for color-aware predictions. |
| **Classifier (baseline)** | KNN (k=15, cosine metric, brute-force) trained alongside LogReg in the local comparison run. Held-out test accuracy ≈ 0.68 vs. LogReg's 0.79 — a **+11 percentage-point lift** with a **45× smaller artifact**. |
| **Food pairing** | Curated `variety → (foods, flavors)` table covering 60+ varieties (95.7% of reviews), plus `cuisine → varieties` and `principle → rule` lookups. Sourced from Wine Folly, Court of Master Sommeliers (US & Europe), MasterClass, The Wine Matchmaker (Antony Anderson, WSET L3), Thermador's Ultimate Wine Pairing Guide, and H-E-B's Wine Pairing Chart. |
| **Origin awareness** | Detects 50+ country and region keywords in the query (German, French, Italian, Bordeaux, Tuscany, Napa, Mosel, Marlborough, Mendoza, etc.) and silently converts them into hard filters on the dataset's `country` and `province` columns. The origin keyword is stripped from the search text so it doesn't pollute the TF-IDF similarity score. |

## Why each design choice

- **TF-IDF over embeddings.** Wine descriptions are short, vocabulary-rich, and domain-specific. TF-IDF beats general-purpose embeddings on this kind of jargon-heavy corpus and trains in seconds on CPU.
- **Cosine similarity, not Euclidean.** TF-IDF vectors are length-sensitive; cosine normalizes for description length.
- **Within-variety value z-score.** Lets us compare "good value Pinot Noir" against "good value Bordeaux" fairly across stylistic categories.
- **Logistic Regression over KNN.** We evaluated both on the same 85/15 split. LogReg outperformed KNN by 11 percentage points on accuracy, produced an artifact 45× smaller (~470 KB vs ~21 MB), and gives calibrated probability outputs we use for color-aware variety prediction. We baselined with KNN as the simplest content-based classifier; the LogReg swap is documented in `artifacts/metrics.txt`.
- **Classifier as query expander, not filter.** A hard filter on predicted variety would be brittle (~79% accuracy ≠ 99%). Using it as a small boost (+0.03 weight on the predicted variety, plus +0.05 when the variety pairs with the user's food keyword) gets the lift without the risk.
- **Hard filters for color, price, and origin.** When the user explicitly picks "white" or types "German wine" or sets a $30 max, they mean it. Filters are applied to the candidate set before similarity ranking.
- **Color-aware variety prediction.** `predict_variety()` consults `predict_proba()` to walk down the ranked classes and return the highest-probability variety that matches the user's color filter — so picking "white" can't return "Cabernet Sauvignon" in the banner.
- **De-duplicate by winery in top-N.** A single producer doesn't dominate the result list.
- **Inflation-adjusted prices.** The Kaggle dataset was scraped in 2017. We apply a 1.33× multiplier (BLS CPI-U, Dec 2017 → 2026 estimate) so the dollar amounts shown reflect roughly current retail. The original column is preserved as `price_2017` for auditability.

## Evaluation

`artifacts/metrics.txt` contains the full side-by-side comparison when `train.py` is run locally with `SKIP_EVAL=False`:

```
top_n_varieties: 12 (+2 must-include: Sparkling Blend, Champagne Blend)
train rows:      65,314
test rows:       11,526
features:        5,000

KNN baseline (k=15, cosine)   accuracy: 0.6800
LogisticRegression (deployed)  accuracy: 0.7900
Improvement                              +11.00 pp
```

Per-class precision/recall/F1 is in the same file. We additionally ran a three-way comparison including LinearSVC, which landed at 0.7833 accuracy — within noise of LogReg. We deployed LogReg for its `predict_proba` outputs (used by color-aware variety prediction) and its native multinomial multiclass formulation.

Recommender quality is best evaluated qualitatively across canonical pairings (e.g. Riesling for Thai curry, Cabernet for ribeye, Sangiovese for tomato pasta, Sparkling Blend for oysters).

## Sample queries that demonstrate the system

```
"light, crisp, citrusy for grilled salmon under $25"   → Sauvignon Blanc / Chardonnay
"bold structured red for a steak dinner"                → Cabernet Sauvignon / Bordeaux blend
"off-dry aromatic for thai green curry"                 → Riesling / Gewürztraminer
"crisp bright bubbly for oysters"                       → Sparkling Blend / Champagne Blend
"German Riesling for sushi"                             → only German Rieslings (Mosel, Rheinhessen)
"Bordeaux for steak"                                    → only French Bordeaux-style blends
"Tuscan wine for pasta"                                 → only Tuscan reds (Sangiovese, Chianti)
"Carmenère with empanadas"                              → Chilean Carmenère
```

## What's stubbed for now

- **Restaurant-menu integration** (a stretch goal from the original proposal). Scaffolded in the recommender but no live scraper is wired up.
- **LLM-based query expansion** for occasion phrases (e.g. "date night wine," "Thanksgiving wine"). Currently handled by curated lookups; an LLM layer would extend coverage to unanticipated phrasings.

## Team

BANA 6920 Machine Learning Applications in Business · Cornell SC Johnson College of Business · Team 5: Alvin Man, Rachel Cheng, Jonathon Gruber, Mark Romeo, Jack Guttenberger.

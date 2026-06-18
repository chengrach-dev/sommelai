# Sommel-AI

A natural-language wine recommendation engine. Tell it what you want in plain English ("light, citrusy white for grilled salmon under $25"), and it returns ranked wine picks from 130,000 Wine Enthusiast reviews — balanced for description match, price-quality value, and traditional food pairings.

## What's inside

```
sommelai/
├── app.py                  # Streamlit demo UI
├── requirements.txt
├── data/
│   ├── winemag-data-130k-v2.csv   # (you provide — Kaggle Wine Reviews)
│   └── wines_clean.parquet         # produced by preprocess.py
├── artifacts/              # produced by train.py
│   ├── tfidf_vectorizer.joblib
│   ├── tfidf_matrix.joblib
│   ├── wines_meta.parquet
│   ├── knn_classifier.joblib
│   ├── knn_labels.joblib
│   └── metrics.txt
└── src/
    ├── preprocess.py       # cleaning + price imputation + value score
    ├── train.py            # TF-IDF + KNN training
    ├── recommender.py      # ranking engine (cosine + value + boosts)
    └── food_pairing.py     # variety → food/flavor knowledge base
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. place the Kaggle dataset at data/winemag-data-130k-v2.csv
#    (or update DEFAULT_INPUTS in src/preprocess.py to point elsewhere)

# 3. build artifacts (~30 seconds)
python src/preprocess.py
python src/train.py

# 4. run the demo
streamlit run app.py
```

## Methodology in one breath

| Layer | What it does |
|---|---|
| **Preprocessing** | Cleans descriptions, imputes missing prices by `(variety, country)` median, computes `value_raw = points / log(1+price)` and z-scores it within variety. |
| **TF-IDF** | `max_features=20k`, `ngram=(1,2)`, `min_df=5`, `sublinear_tf=True` on cleaned descriptions. Produces a 130k × 20k sparse matrix. |
| **Recommender** | `final = α · cosine_sim + (1-α) · value_score`, plus small boosts for (a) varieties whose food pairings match the user's dish and (b) the variety the classifier thinks the query sounds like. |
| **KNN classifier** | k=15, cosine metric, trained on top-25 varieties (~100k rows). Used as a query-expansion signal — predicts which variety your query "sounds like" and adds that variety's flavor/food keywords to the search. |
| **Food pairing** | Hand-curated `variety → (foods, flavors)` table built from Wine Folly, Court of Master Sommeliers, MasterClass references. |

## Why each design choice

- **TF-IDF over embeddings**: descriptions are short, vocabulary-rich, and domain-specific. TF-IDF beats general-purpose embeddings on this kind of jargon-heavy corpus, and trains in seconds on CPU.
- **Cosine similarity, not Euclidean**: TF-IDF vectors are length-sensitive; cosine normalizes for description length.
- **Within-variety value z-score**: lets us compare "good value Pinot Noir" against "good value Bordeaux" fairly.
- **Classifier as query expander, not filter**: a hard filter on predicted variety would be too brittle (57% accuracy ≠ 99%); using it as a small boost (+0.03) gets the lift without the risk.
- **De-duplicate by winery in top-N**: so a single winery doesn't dominate the result list.

## Evaluation

`artifacts/metrics.txt` contains the held-out classifier report. Recommender quality is best evaluated qualitatively + with a small user study (see project doc).

## What's stubbed for now

- **Restaurant-menu integration** (the proposal's stretch goal). Scaffolded in the recommender but no scraper is wired up.
- **Lemmatization**. We rely on TF-IDF's `stop_words="english"` and sublinear TF. Swapping in spaCy/NLTK lemmatization is a one-function change in `src/preprocess.py`.

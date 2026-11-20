# 🎬 Sentiment Analysis Classifier for Movie Reviews

Classic NLP pipeline: **CountVectorizer (bag-of-words) + Logistic Regression**.
Dataset: 2,000 labelled reviews (1,000 positive / 1,000 negative) from the NLTK `movie_reviews` corpus, bundled as `data/movie_reviews.csv`.

## Task steps → code
| # | Step | File |
|---|------|------|
| 1 | Load the movie reviews dataset | `src/data_loader.py` |
| 2 | Clean & preprocess (lowercase, remove punctuation) | `src/preprocess.py` |
| 3 | Bag-of-words with `CountVectorizer` | `src/model.py` → `build_vectorizer` |
| 4 | Train `LogisticRegression` | `src/model.py` → `train_classifier` |
| 5 | Accuracy + confusion matrix | `src/evaluate.py` |

## Run
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py            # saves outputs/confusion_matrix.png + outputs/report.txt
python main.py --show     # also opens the confusion matrix window
pytest -q tests           # unit tests
```

## Result (80/20 stratified split, random_state=42)
Accuracy ≈ **82.5%** — see `outputs/confusion_matrix.png` and `outputs/report.txt`.

## Design notes
- Vectorizer is **fit on the training split only** (no data leakage).
- Stop-words removed, `min_df=2`, vocabulary capped at 20k features.
- `top_features()` prints the words the model weights most for each class.

## Ideas to improve
TF-IDF + n-grams (1,2), `GridSearchCV` over `C`, Naive Bayes / LinearSVC comparison, cross-validation.

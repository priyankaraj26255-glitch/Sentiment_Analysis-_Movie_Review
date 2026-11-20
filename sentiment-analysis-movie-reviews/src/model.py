"""Steps 3 & 4 - Bag-of-words vectorisation and LogisticRegression training."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def build_vectorizer(max_features: int = 20000) -> CountVectorizer:
    return CountVectorizer(stop_words="english", max_features=max_features, min_df=2)


def train_classifier(X_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def top_features(vectorizer, clf, n: int = 10):
    """Most negative / most positive words according to the model."""
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return names[order[:n]].tolist(), names[order[-n:][::-1]].tolist()

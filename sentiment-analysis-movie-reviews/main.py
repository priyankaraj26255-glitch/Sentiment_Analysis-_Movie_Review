"""Sentiment analysis classifier for movie reviews - run the full pipeline."""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def main(show: bool = False):
    if not show:
        matplotlib.use("Agg")  # headless: just save the PNG
    from src.data_loader import load_dataset
    from src.evaluate import evaluate
    from src.model import build_vectorizer, top_features, train_classifier
    from src.preprocess import preprocess_series

    print("[1/5] Loading dataset...")
    df = load_dataset()
    print(f"      {len(df)} reviews | classes: {df['sentiment'].value_counts().to_dict()}")

    print("[2/5] Cleaning text...")
    df["clean"] = preprocess_series(df["review"])

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    print("[3/5] Bag-of-words with CountVectorizer...")
    vec = build_vectorizer()
    X_train = vec.fit_transform(X_train_txt)  # fit on train only -> no leakage
    X_test = vec.transform(X_test_txt)
    print(f"      vocabulary size: {len(vec.vocabulary_)}")

    print("[4/5] Training LogisticRegression...")
    clf = train_classifier(X_train, y_train)

    print("[5/5] Evaluating...")
    acc, cm, report = evaluate(clf, X_test, y_test, ROOT / "outputs" / "confusion_matrix.png", show)
    print(f"\nAccuracy: {acc:.4f}\n\nConfusion matrix:\n{cm}\n\n{report}")

    neg, pos = top_features(vec, clf)
    print("Most negative words:", ", ".join(neg))
    print("Most positive words:", ", ".join(pos))

    (ROOT / "outputs" / "report.txt").write_text(
        f"Accuracy: {acc:.4f}\n\nConfusion matrix:\n{cm}\n\n{report}")

    demos = ["A great, funny and memorable film with an excellent cast.",
             "Boring, predictable and a complete waste of time. Worst script ever."]
    preds = clf.predict(vec.transform(preprocess_series(pd.Series(demos))))
    print("\nDemo predictions:")
    for t, p in zip(demos, preds):
        print(f"  '{t}' -> {'Positive' if p else 'Negative'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="open the confusion matrix window")
    main(ap.parse_args().show)

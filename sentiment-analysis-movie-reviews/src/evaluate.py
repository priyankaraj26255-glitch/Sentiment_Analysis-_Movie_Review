"""Step 5 - Evaluate accuracy and display a confusion matrix."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

LABELS = ["Negative", "Positive"]


def evaluate(clf, X_test, y_test, out_path: Path, show: bool = False):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=LABELS)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (accuracy = {acc:.2%})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return acc, cm, report

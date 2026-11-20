"""Step 1 - Load the movie reviews dataset."""
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "movie_reviews.csv"


def load_dataset(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the CSV (columns: review, sentiment[pos|neg]) and add a 0/1 label."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "pos").astype(int)
    return df

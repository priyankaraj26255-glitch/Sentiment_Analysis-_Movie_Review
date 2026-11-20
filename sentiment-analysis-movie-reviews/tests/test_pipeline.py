import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset
from src.preprocess import clean_text


def test_clean_text_lowercase_and_punct():
    assert clean_text("Great Movie!!! 10/10, wow...") == "great movie wow"


def test_dataset_loads_balanced():
    df = load_dataset()
    assert len(df) == 2000
    assert df["label"].value_counts().nunique() == 1

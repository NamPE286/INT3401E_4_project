from __future__ import annotations

import csv
from pathlib import Path
from utils.string_utils import clean
from tokenizer.basic_tokenizer import tokenize


def load_corpus() -> list[str]:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "imdb.csv"
    reviews: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("review", "")
            if review is not None:
                reviews.extend(clean(review))

    return reviews

def load_tokenized_corpus() -> list[list[str]]:
    tokenized_corpus = []
    
    for i in load_corpus():
        tokenized_corpus.append(tokenize(i))
        
    return tokenized_corpus
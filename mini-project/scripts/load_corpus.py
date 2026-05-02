from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def load_corpus() -> List[str]:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "imdb.csv"
    reviews: List[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row.get("review", "")
            if review is not None:
                reviews.append(review)

    return reviews
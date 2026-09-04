"""Bring your own tabular dataset as a CSV with a target column."""
import sys

import pandas as pd

from beta import BETA

if len(sys.argv) != 3:
    raise SystemExit("usage: python 02_custom_csv.py <csv_path> <target_column>")

csv_path, target_column = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
X, y = df.drop(columns=[target_column]), df[target_column]

model = BETA.from_pretrained()
result = model.recommend(X=X, y=y, dataset_id=csv_path)

print("pipeline:", result.pipeline)
print("selected via:", result.source, "score:", result.score)

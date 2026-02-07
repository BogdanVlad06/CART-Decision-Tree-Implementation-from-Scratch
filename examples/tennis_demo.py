import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from classification_binaryTree import classification_binaryTree

# Load dataset
df = pd.read_csv("examples/datasets/play_tennis.csv")

# Drop non-feature column
df = df.drop(columns=["day"])

# Cast categoricals
for col in df.columns:
    df[col] = df[col].astype("category")

# Split features / target
X = df.drop(columns=["play"])
y = df["play"]

# Train
clf = classification_binaryTree(maxDepth=3)
clf.fit(X, y)

# Inspect tree
clf.print_tree()

# Predict a single example
sample = pd.DataFrame({
    "outlook": ["Rain"],
    "temp": ["Cold"],
    "humidity": ["High"],
    "wind": ["Weak"]
})

for col in sample.columns:
    sample[col] = sample[col].astype("category")

print("Prediction:", clf.predict(sample))

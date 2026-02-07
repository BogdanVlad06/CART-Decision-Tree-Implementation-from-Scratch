# Problem statement: https://judge.nitro-ai.org/competitions/roai-2025/simulare-ojia/1/view

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from classification_binaryTree import classification_binaryTree

# --------------------
# Load data
# --------------------
train_df = pd.read_csv("datasets/train_data.csv")
test_df = pd.read_csv("datasets/test_data.csv")
sample_output = pd.read_csv("datasets/sample_output.csv")

# --------------------
# Prepare training data
# --------------------
X = train_df.drop(columns=["id", "on_time"])
y = train_df["on_time"]

# Convert categorical columns (important for your tree)
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].astype("category")

# --------------------
# Train model
# --------------------
clf = classification_binaryTree(maxDepth=10)
clf.fit(X, y)

# Optional: inspect tree
clf.print_tree()

# --------------------
# Prepare test data
# --------------------
X_test = test_df.drop(columns=["id"])

for col in X_test.select_dtypes(include=["object"]).columns:
    X_test[col] = X_test[col].astype("category")

# --------------------
# Predict
# --------------------
predictions = clf.predict(X_test)

# --------------------
# Build output.csv
# --------------------
output_trimmed = sample_output.iloc[:2][["subtaskID", "datapointID", "answer"]]

output_data = pd.DataFrame({
    "subtaskID": 3,
    "datapointID": test_df["id"],
    "answer": predictions
})

final_output = pd.concat([output_trimmed, output_data], ignore_index=True)
final_output.to_csv("output.csv", index=False)

print("Done. output.csv saved.")
print(final_output.head())

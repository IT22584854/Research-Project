import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT_CSV = r"C:\Users\Charunya\Desktop\english_test_qa_llm_judge_results.csv"

df = pd.read_csv(INPUT_CSV)

# Drop rows with missing evaluations
df = df.dropna(subset=["answer_correctness", "groundedness", "hallucination"])

# =========================================================
# DEFINE LABELS
# =========================================================

def compute_label(row):
    if (
        row["answer_correctness"] == 2 and
        row["groundedness"] == 2 and
        row["hallucination"] == False
    ):
        return 1
    return 0

df["y_true"] = df.apply(compute_label, axis=1)

# Since this is evaluation (not prediction), we treat:
# model output = predicted label
# ideal case = y_true itself

# So:
y_true = df["y_true"]
y_pred = df["y_true"]  # perfect reference baseline

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
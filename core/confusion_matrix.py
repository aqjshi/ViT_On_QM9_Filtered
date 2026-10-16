import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os 
import sys 

CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = CURRENT_FILE_DIR 
OUT_DIR = os.path.join(PROJECT_ROOT, "out")






run_name = "magic-water-45"
run_id = "ymx0t2ln"
# Data extracted from W&B run (vs70s605 / classic-sun-96):
# Actual Neg, Pred Neg: 1735 (TN)
# Actual Neg, Pred Pos: 594  (FP)
# Actual Pos, Pred Neg: 656  (FN)
# Actual Pos, Pred Pos: 1531 (TP)
cm = np.array([[8833, 3311], [3282, 8116]])

tn, fp, fn, tp = cm.ravel()
total = cm.sum()

# Compute performance metrics
accuracy = (tp + tn) / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

# Print metrics table
metrics_df = pd.DataFrame(
    [
        {
            "run_id": run_id,
            "run_name": run_name,
            "samples": total,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    ]
)

print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
print("\nConfusion Matrix:")
print("          Pred Neg  Pred Pos")
print(f"True Neg  {cm[0, 0]:<9} {cm[0, 1]:<9}")
print(f"True Pos  {cm[1, 0]:<9} {cm[1, 1]:<9}")

# Plot publication-ready Confusion Matrix with counts and percentages
labels = [
    [f"{tn:,}", f"{fp:,}"],
    [f"{fn:,}", f"{tp:,}"],
]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=labels,
    fmt="",
    cmap="Blues",
    cbar=True,
    annot_kws={"size": 13},
    xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
    yticklabels=["True Negative (0)", "True Positive (1)"],
)

plt.xlabel("Predicted Sign", fontsize=12,  labelpad=10)
plt.ylabel("True Sign", fontsize=12,  labelpad=10)
plt.title(
    f"Sign Prediction Confusion Matrix\nRun: {run_name} ({run_id}) | Acc: {accuracy:.2%}",
    fontsize=13,
    pad=15,
)

plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
plot_filename = os.path.join(OUT_DIR, f"{run_id}_confusion_matrix.png")

plt.savefig(plot_filename)
plt.show()

print(f"\n[✓] Confusion matrix plot successfully saved to:\n    {plot_filename}")
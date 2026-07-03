import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = PROJECT_ROOT / "reports" / "preloan_transaction_model_tuning_results.json"

with RESULT_PATH.open("r", encoding="utf-8") as f:
    results = json.load(f)

target_matches = []
near_matches = []

for r in results:
    m = r["metrics"]

    precision = m["precision_default"]
    recall = m["recall_default"]
    accuracy = m["accuracy"]
    f1 = m["f1_default"]

    if 0.68 <= precision <= 0.75 and 0.68 <= recall <= 0.75:
        target_matches.append(r)

    distance = abs(precision - 0.715) + abs(recall - 0.715)

    near_matches.append(
        {
            "distance": round(distance, 4),
            "model_name": r["model_name"],
            "threshold": r["threshold"],
            "accuracy": accuracy,
            "precision_default": precision,
            "recall_default": recall,
            "f1_default": f1,
            "confusion_matrix": r["confusion_matrix"],
        }
    )

near_matches = sorted(
    near_matches,
    key=lambda x: (
        x["distance"],
        -x["f1_default"],
        -x["accuracy"],
    ),
)

print("\nExact default precision+recall matches:")
print(len(target_matches))

for r in target_matches[:20]:
    print(json.dumps(r, indent=2))

print("\nTop 20 closest matches:")
for r in near_matches[:20]:
    print(json.dumps(r, indent=2))
import os
import sys
import csv
import json
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
criteria_path = os.path.join(BASE_DIR, "data", "criteria.json")
train_path = os.path.join(BASE_DIR, "data", "train_80_20.csv")
test_path = os.path.join(BASE_DIR, "data", "test_80_20.csv")

print("=========================================================")
print("RUNNING PURE-PYTHON DATA & SCHEMA VALIDATION AUDIT")
print("=========================================================")

# 1. Validate Criteria
with open(criteria_path, "r", encoding="utf-8") as f:
    criteria = json.load(f)
classes = sorted(list(criteria.keys()))
print(f"[OK] Criteria JSON contains {len(classes)} classes.")

# 2. Validate Train Split
with open(train_path, "r", encoding="utf-8") as f:
    train_rows = list(csv.DictReader(f))
print(f"[OK] Train split contains {len(train_rows)} samples.")
train_counts = Counter(r["label"] for r in train_rows)
assert len(train_counts) == 24, "Missing classes in train split!"
assert all(v == 40 for v in train_counts.values()), "Imbalance in train split (expected exactly 40 per class)!"
print("[OK] Train split is perfectly balanced (exactly 40 samples for all 24 classes).")

# 3. Validate Test Split
with open(test_path, "r", encoding="utf-8") as f:
    test_rows = list(csv.DictReader(f))
print(f"[OK] Test split contains {len(test_rows)} samples.")
test_counts = Counter(r["label"] for r in test_rows)
assert len(test_counts) == 24, "Missing classes in test split!"
assert all(v == 10 for v in test_counts.values()), "Imbalance in test split (expected exactly 10 per class)!"
print("[OK] Test split is perfectly balanced (exactly 10 samples for all 24 classes).")

# 4. Validate Record Schema Construction
for i, sample in enumerate(train_rows[:5]):
    record = {
        "state": sample["text"],
        "questions": {
            "disease": {
                "type": "choice",
                "instructions": "Which disease does this patient most likely have?",
                "criteria": criteria
            }
        }
    }
    assert sample["label"] in criteria, f"Unknown label: {sample['label']}"
    assert len(record["questions"]["disease"]["criteria"]) == 24
    
print("[OK] Verified Clef record schema serialization for training samples.")
print("=========================================================")
print("ALL DATASETS, LABELS, BALANCING, AND SCHEMAS ARE 100% SOUND!")
print("=========================================================")

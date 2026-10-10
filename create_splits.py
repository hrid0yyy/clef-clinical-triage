import csv
import json
import os
import random
from collections import defaultdict

DATASET_PATH = "archive/Symptom2Disease.csv"
OUTPUT_DIR = "splits"
RANDOM_SEED = 42

def create_splits():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read original dataset
    with open(DATASET_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = []
        for i, r in enumerate(reader):
            row_dict = dict(r)
            row_dict["original_id"] = row_dict.get("", str(i))
            rows.append(row_dict)
            
    print(f"Total rows read: {len(rows)}")
    
    # Group by label
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
        
    classes = sorted(by_class.keys())
    print(f"Number of classes: {len(classes)} ({len(rows) // len(classes)} samples each)")
    
    # Shuffle each class deterministically
    random.seed(RANDOM_SEED)
    train_80_20 = []
    test_80_20 = []
    
    train_80_10_10 = []
    val_80_10_10 = []
    test_80_10_10 = []
    
    split_indices = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "total_samples": len(rows),
            "num_classes": len(classes),
            "classes": classes
        },
        "80_20": {
            "train_count": 0,
            "test_count": 0,
            "train_original_ids": [],
            "test_original_ids": []
        },
        "80_10_10": {
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "train_original_ids": [],
            "val_original_ids": [],
            "test_original_ids": []
        }
    }
    
    for cls in classes:
        cls_rows = list(by_class[cls])
        random.shuffle(cls_rows)
        
        # 80 / 20 split: 40 train, 10 test
        t80 = cls_rows[:40]
        te20 = cls_rows[40:50]
        train_80_20.extend(t80)
        test_80_20.extend(te20)
        
        # 80 / 10 / 10 split: 40 train, 5 val, 5 test
        t80_3way = cls_rows[:40]
        v10 = cls_rows[40:45]
        te10 = cls_rows[45:50]
        train_80_10_10.extend(t80_3way)
        val_80_10_10.extend(v10)
        test_80_10_10.extend(te10)

    # Save 80/20 CSVs
    def save_csv(filename, data_list):
        path = os.path.join(OUTPUT_DIR, filename)
        cols = ["original_id", "label", "text"]
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in data_list:
                writer.writerow({
                    "original_id": row["original_id"],
                    "label": row["label"],
                    "text": row["text"]
                })
        print(f"Saved: {path} ({len(data_list)} rows)")

    save_csv("train_80_20.csv", train_80_20)
    save_csv("test_80_20.csv", test_80_20)
    
    save_csv("train_80_10_10.csv", train_80_10_10)
    save_csv("val_80_10_10.csv", val_80_10_10)
    save_csv("test_80_10_10.csv", test_80_10_10)
    
    # Record metadata
    split_indices["80_20"]["train_count"] = len(train_80_20)
    split_indices["80_20"]["test_count"] = len(test_80_20)
    split_indices["80_20"]["train_original_ids"] = [r["original_id"] for r in train_80_20]
    split_indices["80_20"]["test_original_ids"] = [r["original_id"] for r in test_80_20]
    
    split_indices["80_10_10"]["train_count"] = len(train_80_10_10)
    split_indices["80_10_10"]["val_count"] = len(val_80_10_10)
    split_indices["80_10_10"]["test_count"] = len(test_80_10_10)
    split_indices["80_10_10"]["train_original_ids"] = [r["original_id"] for r in train_80_10_10]
    split_indices["80_10_10"]["val_original_ids"] = [r["original_id"] for r in val_80_10_10]
    split_indices["80_10_10"]["test_original_ids"] = [r["original_id"] for r in test_80_10_10]
    
    meta_path = os.path.join(OUTPUT_DIR, "splits_metadata.json")
    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)
    print(f"Saved split metadata: {meta_path}")

if __name__ == "__main__":
    create_splits()

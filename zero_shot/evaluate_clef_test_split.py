import csv
import os
import json
import time
import random
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
AUTH_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
API_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef"
TEST_DATA_PATH = "data/test_80_20.csv"
OUTPUT_CSV = "zero_shot/results/clef_test_split_eval_results.csv"
OUTPUT_JSON = "zero_shot/results/clef_test_split_eval_results.json"
MAX_WORKERS = 8
MAX_RETRIES = 5

def load_test_dataset(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for i, row in enumerate(reader):
            data.append({
                "split_idx": i + 1,
                "original_id": row["original_id"],
                "label": row["label"].strip(),
                "text": row["text"].strip()
            })
    return data

def call_clef_api(text, all_labels):
    payload = {
        "model": "clef",
        "state": text,
        "questions": {
            "disease": {
                "type": "choice",
                "instructions": "Which disease does this patient most likely have based on the reported symptoms?",
                "criteria": {label: None for label in all_labels}
            }
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(API_URL, data=data_bytes, headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                elapsed = time.time() - t0
                body = json.loads(resp.read().decode("utf-8"))
                if not body.get("success", False):
                    raise RuntimeError(f"API returned success=False: {body.get('errors')}")
                result = body["result"]
                answer = result["answers"]["disease"]
                usage = result.get("usage", {})
                return {
                    "choice": answer.get("choice"),
                    "confidence": answer.get("confidence", 0.0),
                    "probabilities": answer.get("probabilities", {}),
                    "elapsed_sec": elapsed,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "status": "success"
                }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception) as e:
            last_error = e
            backoff = (1.5 ** attempt) + random.uniform(0.5, 1.0)
            print(f"[Retry {attempt+1}/{MAX_RETRIES}] Error: {e}. Backing off {backoff:.2f}s...")
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")

def evaluate():
    print("=" * 75)
    print("EVALUATING clef (FULL MODEL, ZERO-SHOT) ON 240 TEST CASES")
    print("=" * 75)

    test_data = load_test_dataset(TEST_DATA_PATH)
    all_labels = sorted(list(set(row["label"] for row in test_data)))
    print(f"Total test instances: {len(test_data)} across {len(all_labels)} classes.")
    print(f"Sending requests to @cf/cloudflare/clef using {MAX_WORKERS} workers...")

    start_time = time.time()
    results = []

    def process_item(item):
        api_res = call_clef_api(item["text"], all_labels)
        pred = api_res["choice"]
        gt = item["label"]
        probs = api_res["probabilities"]
        
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top3 = [p[0] for p in sorted_probs[:3]]

        is_correct = (pred == gt)
        is_top3_correct = (gt in top3)

        return {
            "split_idx": item["split_idx"],
            "original_id": item["original_id"],
            "text": item["text"],
            "ground_truth": gt,
            "predicted": pred,
            "is_correct": is_correct,
            "is_top3_correct": is_top3_correct,
            "confidence": api_res["confidence"],
            "pred_prob": probs.get(pred, 0.0),
            "gt_prob": probs.get(gt, 0.0),
            "top3": top3,
            "sorted_probs": sorted_probs,
            "elapsed_sec": api_res["elapsed_sec"],
            "input_tokens": api_res["input_tokens"]
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_item, item): item for item in test_data}
        completed_count = 0
        running_correct = 0

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            if res["is_correct"]:
                running_correct += 1
            current_acc = (running_correct / completed_count) * 100
            print(f"[{completed_count:3d}/{len(test_data)}] True: {res['ground_truth']:<22} | Pred: {res['predicted']:<22} | Correct: {str(res['is_correct']):<5} | Running Acc: {current_acc:5.1f}%")

    results.sort(key=lambda r: r["split_idx"])
    total_time = time.time() - start_time
    total_samples = len(results)

    num_correct = sum(1 for r in results if r["is_correct"])
    num_top3_correct = sum(1 for r in results if r["is_top3_correct"])
    top1_accuracy = (num_correct / total_samples) * 100
    top3_accuracy = (num_top3_correct / total_samples) * 100
    
    latencies = [r["elapsed_sec"] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    total_tokens = sum(r["input_tokens"] for r in results)

    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        class_stats[gt]["support"] += 1
        if r["is_correct"]:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1

    print("\n" + "=" * 75)
    print("FINAL TEST SPLIT RESULTS: clef (FULL MODEL, ZERO-SHOT) on 240 Cases")
    print("=" * 75)
    print(f"Total Test Cases:         {total_samples}")
    print(f"Top-1 Accuracy:           {num_correct}/{total_samples} ({top1_accuracy:.2f}%)")
    print(f"Top-3 Accuracy:           {num_top3_correct}/{total_samples} ({top3_accuracy:.2f}%)")
    print(f"Total Time:               {total_time:.2f}s (Average Latency: {avg_latency:.2f}s/req)")
    print(f"Total Input Tokens:       {total_tokens} (Average: {total_tokens/total_samples:.1f} tokens/req)")
    print("=" * 75)

    print("\nPER-CLASS PERFORMANCE (Each class has exactly 10 test samples):")
    print(f"{'Class Name':<32} | {'Correct':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8}")
    print("-" * 75)
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        support = stats["support"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{cls:<32} | {tp:2d}/10     | {precision*100:6.1f}%    | {recall*100:6.1f}%  | {f1*100:6.1f}%")

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["split_idx", "original_id", "ground_truth", "predicted", "is_correct", "is_top3_correct", "confidence", "pred_prob", "gt_prob", "top1", "top2", "top3", "elapsed_sec", "input_tokens", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "split_idx": r["split_idx"],
                "original_id": r["original_id"],
                "ground_truth": r["ground_truth"],
                "predicted": r["predicted"],
                "is_correct": r["is_correct"],
                "is_top3_correct": r["is_top3_correct"],
                "confidence": f"{r['confidence']:.4f}",
                "pred_prob": f"{r['pred_prob']:.4f}",
                "gt_prob": f"{r['gt_prob']:.4f}",
                "top1": r["top3"][0] if len(r["top3"]) > 0 else "",
                "top2": r["top3"][1] if len(r["top3"]) > 1 else "",
                "top3": r["top3"][2] if len(r["top3"]) > 2 else "",
                "elapsed_sec": f"{r['elapsed_sec']:.2f}",
                "input_tokens": r["input_tokens"],
                "text": r["text"]
            })
    print(f"\nDetailed CSV results saved to: {OUTPUT_CSV}")

    summary_data = {
        "metadata": {
            "model": "clef",
            "test_split": TEST_DATA_PATH,
            "total_samples": total_samples,
            "total_time_seconds": total_time,
            "average_latency_seconds": avg_latency,
            "total_input_tokens": total_tokens
        },
        "metrics": {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "total_correct": num_correct,
            "total_samples": total_samples
        },
        "per_class": {
            cls: {
                "support": class_stats[cls]["support"],
                "correct": class_stats[cls]["tp"],
                "false_positives": class_stats[cls]["fp"],
                "false_negatives": class_stats[cls]["fn"],
                "precision": class_stats[cls]["tp"] / (class_stats[cls]["tp"] + class_stats[cls]["fp"]) if (class_stats[cls]["tp"] + class_stats[cls]["fp"]) > 0 else 0.0,
                "recall": class_stats[cls]["tp"] / (class_stats[cls]["tp"] + class_stats[cls]["fn"]) if (class_stats[cls]["tp"] + class_stats[cls]["fn"]) > 0 else 0.0
            }
            for cls in class_stats
        },
        "results": results
    }
    with open(OUTPUT_JSON, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Full JSON results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    evaluate()

"""Classical baselines on the same 960/240 split used for Clef-Flash.

Hyperparameters are chosen by 5-fold stratified cross-validation on the
training split only; the test split is evaluated once per final model.
Latency is single-record CPU inference (vectorize + predict), averaged
over all 240 test records.
"""
import json
import os
import platform
import time
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "train_80_20.csv")
TEST = os.path.join(ROOT, "data", "test_80_20.csv")
OUT = os.path.join(ROOT, "baselines", "classical_baselines_results.json")
SEED = 42


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def single_record_latency_ms(model, texts):
    model.predict(texts[:5])  # warm-up
    times = []
    for t in texts:
        t0 = time.perf_counter()
        model.predict([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    X_tr, y_tr = train["text"].tolist(), train["label"].tolist()
    X_te, y_te = test["text"].tolist(), test["label"].tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tfidf = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=1)

    configs = {
        "TF-IDF + Linear SVM": (
            Pipeline([("tfidf", TfidfVectorizer(**tfidf)), ("clf", LinearSVC(random_state=SEED))]),
            {"clf__C": [0.1, 0.3, 1, 3, 10]},
        ),
        "TF-IDF + Logistic Regression": (
            Pipeline([("tfidf", TfidfVectorizer(**tfidf)),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [1, 3, 10, 30, 100]},
        ),
        "TF-IDF + Random Forest": (
            Pipeline([("tfidf", TfidfVectorizer(**tfidf)),
                      ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))]),
            {"clf__n_estimators": [300, 500], "clf__max_features": ["sqrt", "log2"]},
        ),
    }

    results = {"train_size": len(X_tr), "test_size": len(X_te), "seed": SEED,
               "cpu": platform.processor(), "models": {}}
    for name, (pipe, grid) in configs.items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        pred = best.predict(X_te)
        k = int(sum(p == t for p, t in zip(pred, y_te)))
        lat_mean, lat_sd = single_record_latency_ms(best, X_te)
        entry = {
            "best_params": search.best_params_,
            "cv_accuracy": round(100 * search.best_score_, 2),
            "test_accuracy": round(100 * accuracy_score(y_te, pred), 2),
            "test_correct": k,
            "test_accuracy_ci95": wilson(k, len(y_te)),
            "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "latency_ms_mean": round(lat_mean, 3),
            "latency_ms_sd": round(lat_sd, 3),
        }
        if "Random Forest" in name:
            # Seed sensitivity with the selected hyperparameters
            accs = []
            for s in range(5):
                rf = best.set_params(clf__random_state=s).fit(X_tr, y_tr)
                accs.append(100 * accuracy_score(y_te, rf.predict(X_te)))
            entry["seed_accuracies"] = [round(a, 2) for a in accs]
        results["models"][name] = entry
        print(name, entry)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", OUT)


if __name__ == "__main__":
    main()

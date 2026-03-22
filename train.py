"""
train.py
--------
Trains and evaluates four classifiers for star constellation prediction:
  1. k-Nearest Neighbours (baseline)
  2. Random Forest
  3. XGBoost (baseline)
  4. Tuned XGBoost (final model — best performance)

Usage:
    python src/train.py

Input:  data/constellations_cleaned.csv
Output: prints metrics; saves confusion matrix plots to results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

os.makedirs("results", exist_ok=True)

FEATURES = ['RA_deg', 'Dec_deg', 'Vmag', 'AbsMag', 'Distance_ly', 'SpectralClass_Missing']
TARGET = 'Constellation'
RANDOM_STATE = 42


def load_data(path="data/constellations_cleaned.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → results/{filename}")


# ── 1. k-Nearest Neighbours ────────────────────────────────────────────────

def train_knn(X_train_res, y_train_res, X_test_scaled, y_test):
    print("\n=== 1. k-Nearest Neighbours (k=7, Manhattan, distance-weighted) ===")
    knn = KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance')
    knn.fit(X_train_res, y_train_res)

    calibrated = CalibratedClassifierCV(knn, method='sigmoid', cv=5)
    calibrated.fit(X_train_res, y_train_res)

    y_train_pred = calibrated.predict(X_train_res)
    print(f"  Train Accuracy : {accuracy_score(y_train_res, y_train_pred):.4f}")
    print(f"  Train Macro F1 : {f1_score(y_train_res, y_train_pred, average='macro'):.4f}")

    y_pred = calibrated.predict(X_test_scaled)
    print(f"  Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test  Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "kNN Confusion Matrix (All Constellations)", "cm_knn.png")
    return calibrated


# ── 2. Random Forest ───────────────────────────────────────────────────────

def train_random_forest(X_train_res, y_train_res, X_test_scaled, y_test):
    print("\n=== 2. Random Forest (600 trees) ===")
    rf = RandomForestClassifier(
        n_estimators=600, max_depth=None,
        class_weight="balanced", random_state=RANDOM_STATE,
        max_features='log2'
    )
    rf.fit(X_train_res, y_train_res)

    y_train_pred = rf.predict(X_train_res)
    print(f"  Train Accuracy : {accuracy_score(y_train_res, y_train_pred):.4f}")
    print(f"  Train Macro F1 : {f1_score(y_train_res, y_train_pred, average='macro'):.4f}")

    y_pred = rf.predict(X_test_scaled)
    print(f"  Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test  Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Random Forest Confusion Matrix (All Constellations)", "cm_rf.png")
    return rf


# ── 3. XGBoost (baseline) ─────────────────────────────────────────────────

def train_xgboost(X, y):
    print("\n=== 3. XGBoost (baseline, label-encoded) ===")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    xgb = XGBClassifier(
        objective='multi:softmax', num_class=len(le.classes_),
        learning_rate=0.1, max_depth=12, n_estimators=800,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    xgb.fit(X_train_res, y_train_res)

    y_train_pred = xgb.predict(X_train_res)
    print(f"  Train Accuracy : {accuracy_score(y_train_res, y_train_pred):.4f}")
    print(f"  Train Macro F1 : {f1_score(y_train_res, y_train_pred, average='macro'):.4f}")

    y_pred = xgb.predict(X_test)
    print(f"  Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test  Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "XGBoost Confusion Matrix (All Constellations)", "cm_xgb.png")
    return xgb, le


# ── 4. Tuned XGBoost (final model) ────────────────────────────────────────

def train_tuned_xgboost(X, y):
    """
    Random search over XGBoost hyperparameters using stratified k-fold CV.
    This is the final, best-performing model (~70% test accuracy).
    """
    print("\n=== 4. Tuned XGBoost (stratified CV random search) ===")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_dist = {
        "n_estimators":     [400, 600, 800],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.03, 0.05, 0.08],
        "subsample":        [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0.0, 0.1, 0.3],
        "reg_lambda":       [1.0, 3.0, 5.0],
        "reg_alpha":        [0.0, 0.5, 1.0],
        "max_delta_step":   [0, 1, 2],
    }

    n_iter, n_splits = 20, 3
    rng = np.random.RandomState(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def sample_params():
        return {k: rng.choice(v) for k, v in param_dist.items()}

    best_score, best_params = -np.inf, None

    for i in range(1, n_iter + 1):
        params = sample_params()
        fold_scores = []
        print(f"\n  Config {i}/{n_iter}: {params}")

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
            X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = XGBClassifier(
                objective='multi:softprob', num_class=len(le.classes_),
                tree_method='hist', eval_metric='mlogloss',
                random_state=RANDOM_STATE, **params
            )
            model.fit(X_tr, y_tr, verbose=False)
            acc = accuracy_score(y_val, model.predict(X_val))
            fold_scores.append(acc)
            print(f"    Fold {fold_idx}/{n_splits} — val acc: {acc:.4f}")

        mean_acc = np.mean(fold_scores)
        print(f"  Mean CV accuracy: {mean_acc:.4f}")
        if mean_acc > best_score:
            best_score, best_params = mean_acc, params
            print("  *** New best config! ***")

    print(f"\nBest CV accuracy : {best_score:.4f}")
    print(f"Best params      : {best_params}")

    # Train final model with best params
    print("\nTraining final model with best params...")
    best_xgb = XGBClassifier(
        objective='multi:softprob', num_class=len(le.classes_),
        tree_method='hist', eval_metric='mlogloss',
        random_state=RANDOM_STATE, **best_params
    )
    best_xgb.fit(X_train_scaled, y_train, verbose=False)

    y_train_pred = best_xgb.predict(X_train_scaled)
    print(f"  Train Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Train Macro F1 : {f1_score(y_train, y_train_pred, average='macro'):.4f}")

    y_pred = best_xgb.predict(X_test_scaled)
    print(f"  Test  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test  Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Tuned XGBoost Confusion Matrix (Final Model)", "cm_tuned_xgb.png")
    return best_xgb, le, scaler


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    X, y = load_data()
    print(f"Loaded dataset: {X.shape[0]} stars, {y.nunique()} constellations")

    # Shared train/test split + SMOTE for kNN and RF
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    train_knn(X_train_res, y_train_res, X_test_scaled, y_test)
    train_random_forest(X_train_res, y_train_res, X_test_scaled, y_test)
    train_xgboost(X, y)
    train_tuned_xgboost(X, y)


if __name__ == "__main__":
    main()

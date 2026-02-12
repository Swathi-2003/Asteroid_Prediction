# classical_models.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_classical_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Ridge Classifier": RidgeClassifier()
    }

    results = {}
    metrics = {}

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        results[name] = acc
        metrics[name] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"{name:15s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    return results, metrics

'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def run_classical_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=3,random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Ridge Classifier": RidgeClassifier()
    }

    results = {}
    metrics = {}

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=100, shuffle=True, random_state=42)

    print("🔁 Running 20-Fold Cross-Validation and Test Evaluation...\n")

    for name, mdl in models.items():
        # --- Cross-validation ---
        cv_scores = cross_val_score(mdl, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # --- Train on full train data, evaluate on test set ---
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        results[name] = acc
        metrics[name] = {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

        print(f"{name:15s} | CV Mean: {cv_mean:.4f} ± {cv_std:.4f} | "
              f"Test Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    return results, metrics
'''
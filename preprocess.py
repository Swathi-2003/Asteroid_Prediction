'''# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Load dataset ---
df = pd.read_csv("Book1.csv")
print(f"Initial shape: {df.shape}")

# Drop unneeded columns
drop_cols = ["id", "spkid", "full_name", "pdes", "name", "prefix"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# Encode categorical columns
if "pha" in df.columns:
    df["pha"] = df["pha"].map({"Y": 1, "N": 0})
if "neo" in df.columns:
    df["neo"] = df["neo"].map({"Y": 1, "N": 0, "Yes": 1, "No": 0})

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Keep numeric columns only
df = df.select_dtypes(include=[np.number])

# Separate features and target
if "pha" not in df.columns:
    raise ValueError("❌ Column 'pha' not found in dataset!")

y = df["pha"]
X = df.drop(columns=["pha"])

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.to_numpy(), test_size=0.2, stratify=y, random_state=42
)

print("\n✅ Preprocessing complete!")
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
print(f"Unique classes: {np.unique(y_train)}")

# --- Expose variables for import ---
__all__ = ["X_train", "X_test", "y_train", "y_test"]

# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def preprocess_data():
    df = pd.read_csv("Book1.csv")
    print(f"Initial shape: {df.shape}")

    # Drop unnecessary columns
    drop_cols = ["id", "spkid", "full_name", "pdes", "name", "prefix"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Encode categorical fields
    if "pha" in df.columns:
        df["pha"] = df["pha"].map({"Y": 1, "N": 0})
    if "neo" in df.columns:
        df["neo"] = df["neo"].map({"Y": 1, "N": 0, "Yes": 1, "No": 0})

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Keep only numeric data
    df = df.select_dtypes(include=[np.number])

    if "pha" not in df.columns:
        raise ValueError("❌ Column 'pha' not found!")

    y = df["pha"].astype(int)
    X = df.drop(columns=["pha"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n✅ Preprocessing complete!")
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    print(f"Unique classes: {y.unique()}")

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()

'''# preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

def vote_feature_ranking(X, y, feature_names, verbose=False):
    """
    Compute importance scores from 9 methods and vote/select top features.
    Returns average normalized importance per feature (higher -> more important).
    """
    n_features = X.shape[1]
    scores = np.zeros((9, n_features), dtype=float)

    # 1 ExtraTrees
    try:
        et = ExtraTreesClassifier(n_estimators=200, random_state=42)
        et.fit(X, y)
        s = np.abs(et.feature_importances_)
        scores[0, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[0, :] = 0

    # 2 RandomForest
    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        s = np.abs(rf.feature_importances_)
        scores[1, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[1, :] = 0

    # 3 AdaBoost
    try:
        ab = AdaBoostClassifier(n_estimators=200, random_state=42)
        ab.fit(X, y)
        s = np.abs(getattr(ab, "feature_importances_", np.zeros(n_features)))
        scores[2, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[2, :] = 0

    # 4 GradientBoosting
    try:
        gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb.fit(X, y)
        s = np.abs(getattr(gb, "feature_importances_", np.zeros(n_features)))
        scores[3, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[3, :] = 0

    # 5 DecisionTree
    try:
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        s = np.abs(getattr(dt, "feature_importances_", np.zeros(n_features)))
        scores[4, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[4, :] = 0

    # 6 IsolationForest (unsupervised): use feature variance in decision_function approximation
    try:
        iso = IsolationForest(n_estimators=200, random_state=42)
        iso.fit(X)
        # fallback: use absolute mean of tree leaf scores per feature by permutation-lite: use std
        s = np.std(X, axis=0)
        s = np.nan_to_num(s)
        scores[5, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[5, :] = 0

    # 7 VarianceThreshold
    try:
        vt = VarianceThreshold()
        vt.fit(X)
        s = np.var(X, axis=0)
        scores[6, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[6, :] = 0

    # 8 SelectKBest (ANOVA F-value)
    try:
        skb = SelectKBest(score_func=f_classif, k=min(10, n_features))
        skb.fit(X, y)
        s = np.nan_to_num(skb.scores_)
        s[np.isnan(s)] = 0.0
        scores[7, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[7, :] = 0

    # 9 RandomForestEnsemble (again but different seed)
    try:
        rfe = RandomForestClassifier(n_estimators=300, random_state=7)
        rfe.fit(X, y)
        s = np.abs(rfe.feature_importances_)
        scores[8, :] = s / (s.sum() + 1e-12)
    except Exception:
        scores[8, :] = 0

    # Average normalized scores across the 9 methods
    avg_scores = scores.mean(axis=0)
    if verbose:
        for i, fname in enumerate(feature_names):
            print(f"{fname}: score={avg_scores[i]:.4f}")
    return avg_scores

def preprocess_and_select(csv_path="Book1.csv", k_features=4, test_size=0.2, random_state=42, verbose=False):
    """
    Preprocess, select top-k features by voting among 9 methods, return train/test subsets.
    returns: X_train, X_test, y_train, y_test, selected_feature_names
    """
    df = pd.read_csv(csv_path)
    if verbose:
        print("Initial shape:", df.shape)
    # Drop identifiers (common columns you listed)
    drop_cols = ["id", "spkid", "full_name", "pdes", "name", "prefix", "orbit_id", "epoch_cal", "tp_cal"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    if "pha" not in df.columns:
        raise ValueError("Column 'pha' not found in dataset.")
    # Map target and NEO
    df["pha"] = df["pha"].map({"Y": 1, "N": 0, 1:1, 0:0})
    if "neo" in df.columns:
        df["neo"] = df["neo"].map({"Y": 1, "N": 0, "Yes": 1, "No": 0, 1:1, 0:0})

    # Separate target
    y = df["pha"].astype(int)
    X = df.drop(columns=["pha"])

    # Numeric imputation for numeric cols
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy="median")
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    # categorical fill
    for col in obj_cols:
        X[col] = X[col].fillna(X[col].mode().iloc[0]).astype(str)

    # For this pipeline we restrict to numeric features (paper used numeric measures)
    X = X.select_dtypes(include=[np.number])
    if verbose:
        print("Numeric feature count:", X.shape[1])

    # Log1p transform for skew (safe)
    X_log = X.copy()
    for col in X_log.columns:
        # avoid transforming extremely negative numbers
        if (X_log[col] >= 0).all():
            X_log[col] = np.log1p(X_log[col])
        else:
            # if negative values exist, shift by min + 1
            mn = X_log[col].min()
            X_log[col] = np.log1p(X_log[col] - mn + 1)

    # Box-Cox where positive
    pt = PowerTransformer(method="box-cox", standardize=False)
    for col in X_log.columns:
        if (X_log[col] > 0).all():
            try:
                X_log[col] = pt.fit_transform(X_log[[col]])
            except Exception:
                pass

    # Final scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_log), columns=X_log.columns)

    # Feature ranking via 9 methods (voting)
    feature_names = X_scaled.columns.tolist()
    avg_scores = vote_feature_ranking(X_scaled.values, y.values, feature_names, verbose=verbose)

    # pick top-k
    idx_sorted = np.argsort(-avg_scores)
    topk_idx = idx_sorted[:k_features]
    selected_cols = [feature_names[i] for i in topk_idx]
    if verbose:
        print("Selected top-k features:", selected_cols)

    X_sel = X_scaled[selected_cols].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y.values.astype(int), test_size=test_size, stratify=y, random_state=random_state
    )
      

    if verbose:
        print("\n✅ Preprocessing complete!")
        print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
        print("Selected features:", selected_cols)

    return X_train, X_test, y_train, y_test, selected_cols

# When run as script, show basic info
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, selected = preprocess_and_select(verbose=True)
    
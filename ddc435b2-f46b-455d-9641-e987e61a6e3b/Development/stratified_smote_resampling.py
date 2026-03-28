import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

# ── 1. Stratified Train/Test Split (80/20) ──────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_train_raw = X.iloc[train_idx].reset_index(drop=True)
X_test       = X.iloc[test_idx].reset_index(drop=True)
y_train_raw  = y.iloc[train_idx].reset_index(drop=True)
y_test       = y.iloc[test_idx].reset_index(drop=True)

print("=" * 55)
print("  CLASS DISTRIBUTION – BEFORE RESAMPLING")
print("=" * 55)
_counts_before = y_train_raw.value_counts().sort_index()
_total_before  = len(y_train_raw)
for cls, cnt in _counts_before.items():
    print(f"  Class {cls}: {cnt:>5} samples  ({cnt/_total_before*100:.2f}%)")
print(f"  Total training samples: {_total_before}")

# ── 2. Manual SMOTE implementation (no imbalanced-learn required) ────────────
rng = np.random.default_rng(42)

def smote_oversample(X_maj, X_min, target_n, k=5):
    """Generate synthetic minority samples via SMOTE interpolation."""
    _k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=_k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)          # shape (n_min, k+1); col 0 = self
    n_to_generate = target_n - len(X_min)
    synthetic = []
    for _ in range(n_to_generate):
        _i   = rng.integers(0, len(X_min))
        _nn  = indices[_i, 1:]                  # exclude self
        _j   = rng.choice(_nn)
        _lam = rng.uniform(0, 1)
        _new = X_min[_i] + _lam * (X_min[_j] - X_min[_i])
        synthetic.append(_new)
    return np.vstack([X_min, np.array(synthetic)])

_X_arr = X_train_raw.values.astype(float)
_y_arr = y_train_raw.values

_majority_cls  = int(_counts_before.idxmax())
_minority_cls  = int(_counts_before.idxmin())
_majority_mask = (_y_arr == _majority_cls)
_minority_mask = (_y_arr == _minority_cls)

_X_maj = _X_arr[_majority_mask]
_X_min = _X_arr[_minority_mask]
_target_n = int(_counts_before.max())

_X_min_resampled = smote_oversample(_X_maj, _X_min, target_n=_target_n)
_y_min_resampled = np.full(len(_X_min_resampled), _minority_cls)

# Combine majority + oversampled minority
_X_combined = np.vstack([_X_maj, _X_min_resampled])
_y_combined = np.concatenate([np.full(len(_X_maj), _majority_cls), _y_min_resampled])

# Shuffle
_shuffle_idx = rng.permutation(len(_X_combined))
X_train_resampled = pd.DataFrame(_X_combined[_shuffle_idx], columns=X_train_raw.columns)
y_train_resampled = pd.Series(_y_combined[_shuffle_idx].astype(int), name=y_train_raw.name)

print("\n" + "=" * 55)
print("  CLASS DISTRIBUTION – AFTER SMOTE RESAMPLING")
print("=" * 55)
_counts_after = y_train_resampled.value_counts().sort_index()
_total_after  = len(y_train_resampled)
for cls, cnt in _counts_after.items():
    print(f"  Class {cls}: {cnt:>5} samples  ({cnt/_total_after*100:.2f}%)")
print(f"  Total training samples: {_total_after}")

print("\n" + "=" * 55)
print("  FINAL SHAPES")
print("=" * 55)
print(f"  X_train_resampled : {X_train_resampled.shape}")
print(f"  y_train_resampled : {y_train_resampled.shape}")
print(f"  X_test            : {X_test.shape}")
print(f"  y_test            : {y_test.shape}")
print(f"  Imbalance ratio   : 1:{int(_counts_after.max()/_counts_after.min())}")

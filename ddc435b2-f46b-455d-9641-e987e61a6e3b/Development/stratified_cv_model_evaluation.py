import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ── Class imbalance ratio ───────────────────────────────────────────────────
_neg = int((y_train_resampled == 0).sum())
_pos = int((y_train_resampled == 1).sum())
_ratio = _neg / _pos
print(f"Resampled training set → Negative: {_neg}, Positive: {_pos}, ratio: {_ratio:.4f}")
print("Note: XGBoost/LightGBM not in environment — using sklearn equivalents:")
print("  XGBoost (scale_pos_weight) → GradientBoostingClassifier + sample_weight per fold")
print("  LightGBM (is_unbalance)    → HistGradientBoostingClassifier (class_weight='balanced')")
print()

# ── Define the 4 models ─────────────────────────────────────────────────────
_models = {
    "XGBoost (GBC)": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    ),
    "LightGBM (HGBC)": HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    ),
}

# ── Stratified 5-Fold CV (manual loop) ─────────────────────────────────────
_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_X_arr = X_train_resampled.values
_y_arr = y_train_resampled.values

_results = []
_fit_models = {}

print("=" * 65)
print("  STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 65)

for _name, _model in _models.items():
    print(f"\n  ▶ Training {_name}...")
    _fold_scores = {m: [] for m in ["roc_auc", "avg_prec", "f1", "precision", "recall"]}
    _last_estimator = None

    for _tr_idx, _val_idx in _skf.split(_X_arr, _y_arr):
        _Xtr, _Xval = _X_arr[_tr_idx], _X_arr[_val_idx]
        _ytr, _yval = _y_arr[_tr_idx], _y_arr[_val_idx]

        # Apply sample_weight for GBC (mimics scale_pos_weight)
        if _name == "XGBoost (GBC)":
            _sw = np.where(_ytr == 1, _ratio, 1.0)
            _model.fit(_Xtr, _ytr, sample_weight=_sw)
        else:
            _model.fit(_Xtr, _ytr)

        _y_prob = _model.predict_proba(_Xval)[:, 1]
        _y_pred = (_y_prob >= 0.5).astype(int)

        _fold_scores["roc_auc"].append(roc_auc_score(_yval, _y_prob))
        _fold_scores["avg_prec"].append(average_precision_score(_yval, _y_prob))
        _fold_scores["f1"].append(f1_score(_yval, _y_pred, zero_division=0))
        _fold_scores["precision"].append(precision_score(_yval, _y_pred, zero_division=0))
        _fold_scores["recall"].append(recall_score(_yval, _y_pred, zero_division=0))
        _last_estimator = _model

    _means = {k: np.mean(v) for k, v in _fold_scores.items()}
    _stds  = {k: np.std(v)  for k, v in _fold_scores.items()}

    _results.append({
        "Model":      _name,
        "AUC-ROC":    _means["roc_auc"],
        "AUC-PR":     _means["avg_prec"],
        "F1":         _means["f1"],
        "Precision":  _means["precision"],
        "Recall":     _means["recall"],
        "AUC-ROC SD": _stds["roc_auc"],
        "F1 SD":      _stds["f1"],
    })
    _fit_models[_name] = _last_estimator

    print(f"    AUC-ROC:   {_means['roc_auc']:.4f} ± {_stds['roc_auc']:.4f}")
    print(f"    AUC-PR:    {_means['avg_prec']:.4f} ± {_stds['avg_prec']:.4f}")
    print(f"    F1:        {_means['f1']:.4f} ± {_stds['f1']:.4f}")
    print(f"    Precision: {_means['precision']:.4f} ± {_stds['precision']:.4f}")
    print(f"    Recall:    {_means['recall']:.4f} ± {_stds['recall']:.4f}")

# ── Build comparison DataFrame ──────────────────────────────────────────────
cv_results_df = pd.DataFrame(_results).set_index("Model")
cv_results_df = cv_results_df.sort_values("AUC-ROC", ascending=False)

print("\n" + "=" * 65)
print("  MODEL COMPARISON TABLE (sorted by AUC-ROC)")
print("=" * 65)
_display_cols = ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall"]
print(cv_results_df[_display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
print("=" * 65)

# ── Select best model by AUC-ROC ────────────────────────────────────────────
_best_name = cv_results_df["AUC-ROC"].idxmax()
best_model = _fit_models[_best_name]

print(f"\n  ✅ Best model: {_best_name}")
print(f"     AUC-ROC = {cv_results_df.loc[_best_name, 'AUC-ROC']:.4f}")
print(f"     AUC-PR  = {cv_results_df.loc[_best_name, 'AUC-PR']:.4f}")
print(f"     F1      = {cv_results_df.loc[_best_name, 'F1']:.4f}")

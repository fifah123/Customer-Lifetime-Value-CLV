from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COL = "target_spend_6m"
REPEAT_COL = "target_repeat_6m"
DROP_COLS = ["customer_unique_id", "future_orders_6m", TARGET_COL, REPEAT_COL]


def load_data() -> pd.DataFrame:
    parquet_path = PROCESSED_DATA_DIR / "clv_dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {parquet_path}. Run src/build_features.py first."
        )
    df = pd.read_parquet(parquet_path)
    df["customer_state"] = df["customer_state"].astype("category")
    return df


def compute_decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute share of actual 6-month revenue captured in top 10% and top 20% predicted cohorts."""
    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    eval_df = eval_df.sort_values("y_pred", ascending=False).reset_index(drop=True)

    total_revenue = eval_df["y_true"].sum()
    if total_revenue <= 0:
        return {"top_10_rev_share": 0.0, "top_20_rev_share": 0.0, "top_10_lift": 0.0}

    n = len(eval_df)
    top_10_rev = eval_df.iloc[: int(0.10 * n)]["y_true"].sum()
    top_20_rev = eval_df.iloc[: int(0.20 * n)]["y_true"].sum()

    top_10_share = top_10_rev / total_revenue
    top_20_share = top_20_rev / total_revenue

    return {
        "top_10_rev_share": top_10_share,
        "top_20_rev_share": top_20_share,
        "top_10_lift": top_10_share / 0.10,
    }


def evaluate_predictions(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | str]:
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    lift_metrics = compute_decile_lift(y_true, y_pred)

    return {
        "model": model_name,
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "spearman_r": round(float(spearman_corr), 4),
        "top_10_rev_share": round(lift_metrics["top_10_rev_share"] * 100, 2),
        "top_20_rev_share": round(lift_metrics["top_20_rev_share"] * 100, 2),
        "top_10_lift_x": round(lift_metrics["top_10_lift"], 2),
    }


def predict_rfm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Heuristic RFM score scaled to average historical 6-month spend."""
    r_score = 1.0 / (test_df["recency_days"] + 1.0)
    f_score = np.log1p(test_df["frequency"])
    m_score = np.log1p(test_df["monetary_total"])

    raw_score = (
        (r_score / r_score.mean())
        + (f_score / f_score.mean())
        + (m_score / m_score.mean())
    )
    scale_factor = train_df[TARGET_COL].mean() / raw_score.mean()
    return (raw_score * scale_factor).to_numpy()


def train_and_evaluate() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y_spend = df[TARGET_COL].to_numpy()
    y_repeat = df[REPEAT_COL].to_numpy()

    # Stratified split on repeat flag so train/test preserve exact ~1.18% repeat rate
    X_train, X_test, y_spend_train, y_spend_test, y_rep_train, y_rep_test, idx_train, idx_test = (
        train_test_split(
            X,
            y_spend,
            y_repeat,
            df.index,
            test_size=0.20,
            random_state=42,
            stratify=y_repeat,
        )
    )

    print(f"Train cohort: {len(X_train):,} customers | Test cohort: {len(X_test):,} customers")
    print(f"Test repeat buyers: {y_rep_test.sum():,} ({y_rep_test.mean():.2%})\n")

    results = []

    # ---------------------------------------------------------
    # 1. Baseline: RFM Heuristic
    # ---------------------------------------------------------
    rfm_preds = predict_rfm_baseline(df.loc[idx_train], df.loc[idx_test])
    results.append(evaluate_predictions("1. RFM Heuristic Baseline", y_spend_test, rfm_preds))

    # ---------------------------------------------------------
    # 2. Single-Stage XGBoost Tweedie Regressor
    # ---------------------------------------------------------
    tweedie_model = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    tweedie_model.fit(X_train, y_spend_train)
    tweedie_preds = np.clip(tweedie_model.predict(X_test), 0, None)
    results.append(evaluate_predictions("2. XGBoost Tweedie (1-Stage)", y_spend_test, tweedie_preds))

    # ---------------------------------------------------------
    # 3. Two-Stage Hurdle Model (Uncalibrated Classifier + Conditional Regressor)
    # ---------------------------------------------------------
    # Stage 3A: Repeat Purchase Probability P(Return)
    hurdle_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    hurdle_clf.fit(X_train, y_rep_train)
    prob_return_test = hurdle_clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_rep_test, prob_return_test)
    pr_auc = average_precision_score(y_rep_test, prob_return_test)
    print(f"Hurdle Stage 1 (Classifier) -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Stage 3B: Conditional Spend E[Spend | Return = 1] trained on log1p(spend)
    repeat_mask_train = y_rep_train == 1
    X_train_rep = X_train.loc[repeat_mask_train]
    y_log_spend_train = np.log1p(y_spend_train[repeat_mask_train])

    cond_reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=10.0,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    cond_reg.fit(X_train_rep, y_log_spend_train)

    # Apply smearing correction factor to reduce log back-transformation bias
    train_log_preds = cond_reg.predict(X_train_rep)
    smearing_factor = np.mean(np.exp(y_log_spend_train - train_log_preds))
    cond_spend_test = np.expm1(cond_reg.predict(X_test)) * smearing_factor
    cond_spend_test = np.clip(cond_spend_test, 0, None)

    hurdle_preds = prob_return_test * cond_spend_test
    results.append(evaluate_predictions("3. Two-Stage Hurdle XGBoost", y_spend_test, hurdle_preds))

    # ---------------------------------------------------------
    # Save Comparison Summary & Artifacts
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(results)
    print("\n=== 6-Month CLV Model Benchmark (Test Set) ===")
    print(summary_df.to_string(index=False))

    summary_path = PROCESSED_DATA_DIR / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save test-set predictions for downstream Streamlit simulator & SHAP analysis
    test_output = df.loc[idx_test].copy()
    test_output["pred_rfm_clv"] = rfm_preds
    test_output["pred_tweedie_clv"] = tweedie_preds
    test_output["pred_prob_return"] = prob_return_test
    test_output["pred_cond_spend"] = cond_spend_test
    test_output["pred_hurdle_clv"] = hurdle_preds

    preds_path = PROCESSED_DATA_DIR / "test_predictions.parquet"
    test_output.to_parquet(preds_path, index=False)

    # Persist trained models
    joblib.dump(hurdle_clf, MODELS_DIR / "hurdle_classifier.joblib")
    joblib.dump(cond_reg, MODELS_DIR / "hurdle_regressor.joblib")
    joblib.dump(tweedie_model, MODELS_DIR / "tweedie_regressor.joblib")

    print(f"\nSaved benchmark table -> {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved test predictions -> {preds_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved model artifacts  -> {MODELS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    train_and_evaluate()
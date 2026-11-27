from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

DROP_COLS = [
    "customer_unique_id",
    "future_orders_6m",
    "target_spend_6m",
    "target_repeat_6m",
    "pred_rfm_clv",
    "pred_tweedie_clv",
    "pred_prob_return",
    "pred_cond_spend",
    "pred_hurdle_clv",
]


def plot_cumulative_gain_and_deciles(df: pd.DataFrame) -> None:
    """Plot Cumulative Revenue Gain Curve and Decile Lift bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    models = {
        "Two-Stage Hurdle XGBoost": ("pred_hurdle_clv", "#1f77b4", "-"),
        "Single-Stage Tweedie": ("pred_tweedie_clv", "#ff7f0e", "--"),
        "RFM Heuristic Baseline": ("pred_rfm_clv", "#7f7f7f", ":"),
    }

    total_rev = df["target_spend_6m"].sum()
    n = len(df)
    pct_customers = np.linspace(0, 100, n + 1)

    # Panel 1: Cumulative Gain Curve
    ax1 = axes[0]
    for label, (col, color, ls) in models.items():
        sorted_rev = df.sort_values(col, ascending=False)["target_spend_6m"].to_numpy()
        cum_rev_pct = np.insert(np.cumsum(sorted_rev) / total_rev * 100, 0, 0.0)
        ax1.plot(pct_customers, cum_rev_pct, label=label, color=color, linestyle=ls, linewidth=2.2)

    ax1.plot([0, 100], [0, 100], color="black", linestyle="-.", alpha=0.4, label="Random Targeting")
    ax1.axvline(20, color="#d62728", linestyle="--", alpha=0.5)
    ax1.set_title("Cumulative 6-Month Revenue Gain Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("% of Customers Targeted (Ranked by Predicted CLV)")
    ax1.set_ylabel("% of Actual 6-Month Revenue Captured")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, alpha=0.25)

    # Panel 2: Revenue Share by Predicted Decile (Hurdle Model)
    ax2 = axes[1]
    df_sorted = df.sort_values("pred_hurdle_clv", ascending=False).reset_index(drop=True)
    df_sorted["decile"] = pd.qcut(df_sorted.index, q=10, labels=[f"D{i}" for i in range(1, 11)])
    decile_share = (
        df_sorted.groupby("decile", observed=False)["target_spend_6m"].sum() / total_rev * 100
    )

    bars = ax2.bar(decile_share.index, decile_share.values, color="#1f77b4", alpha=0.88)
    ax2.axhline(10.0, color="#d62728", linestyle="--", linewidth=1.2, label="Random Baseline (10%)")
    ax2.set_title("Actual Revenue Share by Predicted CLV Decile (Hurdle)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted CLV Decile (D1 = Top 10%)")
    ax2.set_ylabel("% of Total Actual Revenue")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.4,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    plt.tight_layout()
    out_path = FIGURES_DIR / "01_clv_decile_lift.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved -> {out_path.relative_to(PROJECT_ROOT)}")


def plot_shap_drivers(df: pd.DataFrame) -> None:
    """Generate SHAP summary plots for Stage 1 (Return Probability) and Stage 2 (Conditional Spend)."""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_sample = df[feature_cols].copy()
    if "customer_state" in X_sample.columns:
        X_sample["customer_state"] = X_sample["customer_state"].astype("category")

    X_shap = X_sample.sample(n=min(2000, len(X_sample)), random_state=42)
    dmatrix_shap = xgb.DMatrix(X_shap, enable_categorical=True)

    # Stage 1 Classifier SHAP (native TreeSHAP -> exclude bias column)
    hurdle_clf = joblib.load(MODELS_DIR / "hurdle_classifier.joblib")
    shap_values_clf = hurdle_clf.get_booster().predict(dmatrix_shap, pred_contribs=True)[:, :-1]

    plt.figure(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values_clf, X_shap, max_display=12, show=False)
    plt.title("Stage 1: Drivers of Repeat Purchase Probability P(Return)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_clf = FIGURES_DIR / "02_shap_repeat_probability.png"
    plt.savefig(out_clf)
    plt.close()
    print(f"Saved -> {out_clf.relative_to(PROJECT_ROOT)}")

    # Stage 2 Conditional Spend Regressor SHAP
    cond_reg = joblib.load(MODELS_DIR / "hurdle_regressor.joblib")
    shap_values_reg = cond_reg.get_booster().predict(dmatrix_shap, pred_contribs=True)[:, :-1]

    plt.figure(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values_reg, X_shap, max_display=12, show=False)
    plt.title("Stage 2: Drivers of Conditional Order Spend E[Spend | Return]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_reg = FIGURES_DIR / "03_shap_conditional_spend.png"
    plt.savefig(out_reg)
    plt.close()
    print(f"Saved -> {out_reg.relative_to(PROJECT_ROOT)}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = PROCESSED_DATA_DIR / "test_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError("Run src/train_evaluate.py before generating visualizations.")

    df = pd.read_parquet(preds_path)
    plot_cumulative_gain_and_deciles(df)
    plot_shap_drivers(df)


if __name__ == "__main__":
    main()
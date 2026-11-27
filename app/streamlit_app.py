from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Olist CLV & Retention ROI Simulator",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DATA_DIR / "test_predictions.parquet")

    med_spend = df["monetary_total"].quantile(0.75)
    med_prob = df["pred_prob_return"].quantile(0.75)

    conditions = [
        (df["monetary_total"] >= med_spend) & (df["pred_prob_return"] >= med_prob),
        (df["monetary_total"] >= med_spend) & (df["pred_prob_return"] < med_prob),
        (df["monetary_total"] < med_spend) & (df["pred_prob_return"] >= med_prob),
    ]
    choices = [
        "1. VIP Champions (High Spend, High Return)",
        "2. At-Risk Whales (High Spend, Low Return)",
        "3. Emerging Loyalists (Low Spend, High Return)",
    ]
    df["segment"] = np.select(conditions, choices, default="4. One-Time / Low-Priority")
    return df


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "model_comparison.csv")


def filter_top_pct_per_segment(df: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """Keep the top X% highest predicted CLV customers within each selected segment."""
    if top_pct >= 1.0:
        return df.copy()

    filtered_parts = []
    for _, group in df.groupby("segment", observed=False):
        n_keep = max(1, int(np.ceil(len(group) * top_pct)))
        top_group = group.sort_values("pred_hurdle_clv", ascending=False).head(n_keep)
        filtered_parts.append(top_group)

    return pd.concat(filtered_parts, ignore_index=True) if filtered_parts else df.iloc[0:0].copy()


def main() -> None:
    st.title("Olist E-Commerce: Customer Lifetime Value & Retention ROI")
    st.markdown(
        "Two-Stage Hurdle Model predicting **6-month repeat purchase probability** and "
        "**conditional spend** across Brazilian e-commerce cohorts."
    )

    try:
        df = load_predictions()
        benchmark_df = load_benchmark()
    except FileNotFoundError:
        st.error(
            "Processed predictions not found. Run `python src/build_features.py` and "
            "`python src/train_evaluate.py` first."
        )
        return

    # ---------------------------------------------------------
    # Sidebar: Retention Campaign Controls
    # ---------------------------------------------------------
    st.sidebar.header("Retention Campaign Simulator")

    selected_segments = st.sidebar.multiselect(
        "1. Target Customer Segments",
        options=sorted(df["segment"].unique()),
        default=[
            "1. VIP Champions (High Spend, High Return)",
            "2. At-Risk Whales (High Spend, Low Return)",
        ],
    )

    top_pct_within_seg = (
        st.sidebar.slider(
            "2. Top % of Customers Within Each Segment",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Ranks customers by predicted 6M CLV inside each segment and targets only the top slice.",
        )
        / 100.0
    )

    only_profitable_individuals = st.sidebar.checkbox(
        "Target only positive-ROI individuals (Incremental GP > Cost)",
        value=False,
        help="Filters out individual customers whose expected incremental gross profit is below the campaign cost.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Campaign Economics")

    voucher_cost = st.sidebar.slider(
        "Campaign / Voucher Cost per Customer (R$)",
        min_value=0.5,
        max_value=30.0,
        value=3.0,
        step=0.5,
        help="e.g., R$ 0.50–2.00 for CRM email/push notification, R$ 10–25 for discount voucher.",
    )
    gross_margin = st.sidebar.slider("Product Gross Margin (%)", 10, 80, 40, 5) / 100.0
    relative_lift = (
        st.sidebar.slider(
            "Expected Relative Lift in Return Probability (%)",
            min_value=5,
            max_value=150,
            value=30,
            step=5,
        )
        / 100.0
    )

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Cohort Customers", f"{len(df):,}")
    col2.metric("Baseline Repeat Rate", f"{df['target_repeat_6m'].mean():.2%}")
    col3.metric("Predicted 6M Revenue", f"R$ {df['pred_hurdle_clv'].sum():,.0f}")
    col4.metric("Actual 6M Revenue", f"R$ {df['target_spend_6m'].sum():,.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Retention ROI Simulator", "Customer Segments & Churn Drivers", "Model Benchmark"]
    )

    # ---------------------------------------------------------
    # TAB 1: Campaign ROI Simulator
    # ---------------------------------------------------------
    with tab1:
        seg_subset = df[df["segment"].isin(selected_segments)].copy()

        if seg_subset.empty:
            st.warning("Select at least one customer segment in the sidebar.")
        else:
            targeted = filter_top_pct_per_segment(seg_subset, top_pct_within_seg)

            # Calculate expected lift & unit economics per customer
            new_prob = np.clip(targeted["pred_prob_return"] * (1.0 + relative_lift), 0, 1.0)
            incremental_prob = new_prob - targeted["pred_prob_return"]

            targeted["incremental_rev"] = incremental_prob * targeted["pred_cond_spend"]
            targeted["incremental_gp"] = targeted["incremental_rev"] * gross_margin
            targeted["net_value"] = targeted["incremental_gp"] - voucher_cost

            if only_profitable_individuals:
                targeted = targeted[targeted["net_value"] > 0].copy()

            total_eligible = len(seg_subset)
            total_targeted = len(targeted)
            coverage_pct = (total_targeted / total_eligible * 100) if total_eligible > 0 else 0.0

            incremental_gross_profit = targeted["incremental_gp"].sum()
            total_campaign_cost = total_targeted * voucher_cost
            net_profit = incremental_gross_profit - total_campaign_cost
            roi_pct = (net_profit / total_campaign_cost * 100) if total_campaign_cost > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Targeted Customers",
                f"{total_targeted:,} / {total_eligible:,}",
                delta=f"{coverage_pct:.1f}% of selected tiers",
                delta_color="off",
            )
            m2.metric("Campaign Cost", f"R$ {total_campaign_cost:,.0f}")
            m3.metric("Incremental Gross Profit", f"R$ {incremental_gross_profit:,.0f}")
            m4.metric(
                "Projected Net Profit (ROI)",
                f"R$ {net_profit:,.0f}",
                delta=f"{roi_pct:.1f}% ROI",
            )

            st.subheader("Segment-Level Unit Economics Breakdown")

            # Build full comparison across all segments using the same top-% rule
            all_top = filter_top_pct_per_segment(df, top_pct_within_seg)
            all_new_prob = np.clip(all_top["pred_prob_return"] * (1.0 + relative_lift), 0, 1.0)
            all_top["incremental_gp"] = (
                (all_new_prob - all_top["pred_prob_return"])
                * all_top["pred_cond_spend"]
                * gross_margin
            )
            if only_profitable_individuals:
                all_top = all_top[all_top["incremental_gp"] > voucher_cost]

            all_top["net_value"] = all_top["incremental_gp"] - voucher_cost

            seg_counts = df.groupby("segment", as_index=False).agg(
                total_in_segment=("customer_unique_id", "count")
            )

            seg_summary = (
                all_top.groupby("segment", as_index=False)
                .agg(
                    targeted_customers=("customer_unique_id", "count"),
                    avg_pred_clv=("pred_hurdle_clv", "mean"),
                    avg_return_prob=("pred_prob_return", "mean"),
                    breakeven_cost_per_cust=("incremental_gp", "mean"),
                    total_cost=("net_value", lambda s: len(s) * voucher_cost),
                    incremental_gp=("incremental_gp", "sum"),
                    net_campaign_profit=("net_value", "sum"),
                )
                .merge(seg_counts, on="segment", how="right")
                .fillna(0)
            )
            seg_summary["targeted_pct"] = (
                seg_summary["targeted_customers"] / seg_summary["total_in_segment"] * 100
            )
            seg_summary["is_selected"] = seg_summary["segment"].isin(selected_segments)

            display_cols = [
                "is_selected",
                "segment",
                "total_in_segment",
                "targeted_customers",
                "targeted_pct",
                "avg_pred_clv",
                "breakeven_cost_per_cust",
                "total_cost",
                "incremental_gp",
                "net_campaign_profit",
            ]
            st.dataframe(
                seg_summary[display_cols].sort_values("net_campaign_profit", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "is_selected": st.column_config.CheckboxColumn("Active"),
                    "segment": "Segment",
                    "total_in_segment": st.column_config.NumberColumn("Total Customers", format="%d"),
                    "targeted_customers": st.column_config.NumberColumn("Targeted", format="%d"),
                    "targeted_pct": st.column_config.NumberColumn("Targeted %", format="%.1f%%"),
                    "avg_pred_clv": st.column_config.NumberColumn("Avg Pred CLV", format="R$ %.2f"),
                    "breakeven_cost_per_cust": st.column_config.NumberColumn(
                        "Max Break-Even Cost", format="R$ %.2f"
                    ),
                    "total_cost": st.column_config.NumberColumn("Campaign Cost", format="R$ %.0f"),
                    "incremental_gp": st.column_config.NumberColumn("Incr. Gross Profit", format="R$ %.0f"),
                    "net_campaign_profit": st.column_config.NumberColumn("Net Profit", format="R$ %.0f"),
                },
            )

    # ---------------------------------------------------------
    # TAB 2: Segment Profile & Operational Impact
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Customer Segment Summary (Historical vs. Predicted)")
        profile_df = (
            df.groupby("segment", as_index=False)
            .agg(
                customers=("customer_unique_id", "count"),
                hist_spend_avg=("monetary_total", "mean"),
                avg_review_score=("avg_review_score", "mean"),
                late_delivery_pct=("late_delivery_rate", "mean"),
                pred_return_prob=("pred_prob_return", "mean"),
                pred_6m_clv=("pred_hurdle_clv", "mean"),
                actual_6m_clv=("target_spend_6m", "mean"),
            )
        )
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        st.subheader("Top 25 Highest Predicted CLV Customers")
        top_customers = df.sort_values("pred_hurdle_clv", ascending=False).head(25)[
            [
                "customer_unique_id",
                "customer_state",
                "segment",
                "frequency",
                "monetary_total",
                "avg_review_score",
                "avg_delivery_delay_days",
                "pred_prob_return",
                "pred_cond_spend",
                "pred_hurdle_clv",
                "target_spend_6m",
            ]
        ]
        st.dataframe(top_customers, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 3: Benchmark Table
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Out-of-Sample Model Comparison (Test Set)")
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
        st.caption(
            "Note: Because ~97% of Olist customers have R$ 0 spend in the 6-month target window, "
            "ranking accuracy (Spearman R and Top-10% Revenue Lift) is the primary business metric."
        )


if __name__ == "__main__":
    main()
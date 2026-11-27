from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Temporal split boundaries
OBS_START = "2017-01-01 00:00:00"
CUTOFF_DATE = "2018-02-28 23:59:59"
PRED_START = "2018-03-01 00:00:00"
PRED_END = "2018-08-31 23:59:59"


def build_feature_dataset() -> pd.DataFrame:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    query = f"""
    WITH clean_orders AS (
        SELECT
            o.order_id,
            c.customer_unique_id,
            c.customer_state,
            CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
            CAST(o.order_delivered_customer_date AS TIMESTAMP) AS delivered_ts,
            CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS estimated_ts
        FROM read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_orders_dataset.csv') o
        JOIN read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_customers_dataset.csv') c
            ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
          AND o.order_purchase_timestamp IS NOT NULL
    ),

    -- 1. Pre-aggregate payments at order_id level to avoid cartesian multiplication
    order_payments_agg AS (
        SELECT
            order_id,
            SUM(payment_value) AS order_payment_value,
            MAX(payment_installments) AS max_installments,
            FIRST(payment_type ORDER BY payment_value DESC) AS primary_payment_type,
            COUNT(*) AS payment_methods_count
        FROM read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_order_payments_dataset.csv')
        GROUP BY order_id
    ),

    -- 2. Pre-aggregate items & product categories at order_id level
    order_items_agg AS (
        SELECT
            oi.order_id,
            COUNT(*) AS item_count,
            SUM(oi.price) AS items_price_total,
            SUM(oi.freight_value) AS freight_total,
            COUNT(DISTINCT COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')) AS distinct_categories
        FROM read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_order_items_dataset.csv') oi
        LEFT JOIN read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_products_dataset.csv') p
            ON oi.product_id = p.product_id
        LEFT JOIN read_csv_auto('{RAW_DATA_DIR.as_posix()}/product_category_name_translation.csv') t
            ON p.product_category_name = t.product_category_name
        GROUP BY oi.order_id
    ),

    -- 3. Pre-aggregate review scores at order_id level
    order_reviews_agg AS (
        SELECT
            order_id,
            AVG(review_score) AS review_score
        FROM read_csv_auto('{RAW_DATA_DIR.as_posix()}/olist_order_reviews_dataset.csv')
        GROUP BY order_id
    ),

    -- Combine enriched orders
    enriched_orders AS (
        SELECT
            o.order_id,
            o.customer_unique_id,
            o.customer_state,
            o.purchase_ts,
            DATE_DIFF('day', o.estimated_ts, o.delivered_ts) AS delivery_delay_days,
            DATE_DIFF('day', o.purchase_ts, o.delivered_ts) AS actual_shipping_days,
            COALESCE(p.order_payment_value, 0) AS payment_value,
            COALESCE(p.max_installments, 1) AS max_installments,
            COALESCE(p.primary_payment_type, 'unknown') AS primary_payment_type,
            COALESCE(i.item_count, 1) AS item_count,
            COALESCE(i.items_price_total, 0) AS items_price_total,
            COALESCE(i.freight_total, 0) AS freight_total,
            COALESCE(i.distinct_categories, 1) AS distinct_categories,
            r.review_score
        FROM clean_orders o
        LEFT JOIN order_payments_agg p ON o.order_id = p.order_id
        LEFT JOIN order_items_agg i ON o.order_id = i.order_id
        LEFT JOIN order_reviews_agg r ON o.order_id = r.order_id
    ),

    -- 4. Observation Window Features (X)
    obs_features AS (
        SELECT
            customer_unique_id,
            ANY_VALUE(customer_state) AS customer_state,

            -- RFM Core
            DATE_DIFF('day', MAX(purchase_ts), TIMESTAMP '{CUTOFF_DATE}') AS recency_days,
            DATE_DIFF('day', MIN(purchase_ts), TIMESTAMP '{CUTOFF_DATE}') AS tenure_days,
            COUNT(DISTINCT order_id) AS frequency,
            SUM(payment_value) AS monetary_total,
            AVG(payment_value) AS monetary_avg,
            MAX(payment_value) AS monetary_max,

            -- Inter-purchase cadence (0 if single order)
            CASE
                WHEN COUNT(DISTINCT order_id) > 1
                THEN DATE_DIFF('day', MIN(purchase_ts), MAX(purchase_ts)) * 1.0 / (COUNT(DISTINCT order_id) - 1)
                ELSE -1.0
            END AS avg_days_between_orders,

            -- Operational & Experience Signals
            AVG(review_score) AS avg_review_score,
            MIN(review_score) AS min_review_score,
            AVG(delivery_delay_days) AS avg_delivery_delay_days,
            MAX(delivery_delay_days) AS max_delivery_delay_days,
            SUM(CASE WHEN delivery_delay_days > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS late_delivery_rate,
            AVG(actual_shipping_days) AS avg_shipping_days,

            -- Basket & Payment Signals
            SUM(item_count) AS total_items,
            AVG(item_count) AS avg_items_per_order,
            SUM(freight_total) / NULLIF(SUM(items_price_total), 0) AS freight_to_price_ratio,
            MAX(max_installments) AS max_installments,
            AVG(max_installments) AS avg_installments,
            SUM(distinct_categories) AS total_category_touches,
            SUM(CASE WHEN primary_payment_type = 'credit_card' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS credit_card_share,
            SUM(CASE WHEN primary_payment_type = 'boleto' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS boleto_share
        FROM enriched_orders
        WHERE purchase_ts BETWEEN TIMESTAMP '{OBS_START}' AND TIMESTAMP '{CUTOFF_DATE}'
        GROUP BY customer_unique_id
    ),

    -- 5. Prediction Window Targets (y)
    future_targets AS (
        SELECT
            customer_unique_id,
            COUNT(DISTINCT order_id) AS future_orders_6m,
            SUM(payment_value) AS target_spend_6m
        FROM enriched_orders
        WHERE purchase_ts BETWEEN TIMESTAMP '{PRED_START}' AND TIMESTAMP '{PRED_END}'
        GROUP BY customer_unique_id
    )

    SELECT
        obs.*,
        COALESCE(fut.future_orders_6m, 0) AS future_orders_6m,
        COALESCE(fut.target_spend_6m, 0.0) AS target_spend_6m,
        CASE WHEN COALESCE(fut.target_spend_6m, 0.0) > 0 THEN 1 ELSE 0 END AS target_repeat_6m
    FROM obs_features obs
    LEFT JOIN future_targets fut
        ON obs.customer_unique_id = fut.customer_unique_id
    ORDER BY obs.customer_unique_id;
    """

    print("Running DuckDB SQL feature engineering pipeline...")
    df = con.execute(query).df()

    # Fill missing review scores with median and flag missingness
    df["missing_review_flag"] = df["avg_review_score"].isna().astype(int)
    median_review = df["avg_review_score"].median()
    df["avg_review_score"] = df["avg_review_score"].fillna(median_review)
    df["min_review_score"] = df["min_review_score"].fillna(median_review)
    df["freight_to_price_ratio"] = df["freight_to_price_ratio"].fillna(0.0)

    parquet_path = PROCESSED_DATA_DIR / "clv_dataset.parquet"
    csv_path = PROCESSED_DATA_DIR / "clv_dataset.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"Saved feature dataset -> {parquet_path.relative_to(PROJECT_ROOT)}")
    print(f"Total cohort customers (Obs Window): {len(df):,}")
    print(f"Repeat buyers in 6m target window:   {df['target_repeat_6m'].sum():,} ({df['target_repeat_6m'].mean():.2%})")
    print(f"Total 6m target revenue:             R$ {df['target_spend_6m'].sum():,.2f}")

    return df


if __name__ == "__main__":
    build_feature_dataset()
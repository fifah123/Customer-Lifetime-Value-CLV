from pathlib import Path
import shutil
import kagglehub

# Resolve project root reliably (assuming script sits in olist_clv/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def download_olist_data() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Olist Brazilian E-Commerce dataset via kagglehub...")

    cache_path = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    print(f"Downloaded to cache: {cache_path}")

    copied_count = 0
    for csv_file in cache_path.glob("*.csv"):
        dest_path = RAW_DATA_DIR / csv_file.name
        shutil.copy2(csv_file, dest_path)
        copied_count += 1
        print(f"  Copied -> {dest_path.relative_to(PROJECT_ROOT)}")

    missing = set(EXPECTED_FILES) - {f.name for f in RAW_DATA_DIR.glob("*.csv")}
    if missing:
        raise FileNotFoundError(f"Missing expected dataset files: {missing}")

    print(f"\nSuccess: {copied_count} CSV files ready in {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_olist_data()
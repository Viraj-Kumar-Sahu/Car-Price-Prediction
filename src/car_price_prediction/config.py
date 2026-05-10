from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "cardekho.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / "model.joblib"

TARGET_COLUMN = "selling_price"
REQUIRED_COLUMNS = {
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
}

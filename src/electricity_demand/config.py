from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "time_series_60min_singleindex.csv"

START_DATE = "2015-01-01"

COUNTRY_CODE = "DE"

TARGET_COLUMN = "load_actual_entsoe_transparency"

DATETIME_COLUMN = "utc_timestamp"

DAILY_FREQUENCY = "D"
WEEKLY_FREQUENCY = "W"
HOURLY_FREQUENCY = "H"

TEST_YEARS = 2

RANDOM_STATE = 42

LSTM_SEQUENCE_LENGTH = 24

LSTM_BATCH_SIZE = 32

LSTM_EPOCHS = 50

LSTM_VALIDATION_SPLIT = 0.2

SARIMA_P = range(0, 7)
SARIMA_D = range(0, 3)
SARIMA_Q = range(0, 7)

SARIMA_SEASONAL_PERIOD = 52

RANDOM_FOREST_ESTIMATORS = 300

GRADIENT_BOOSTING_ESTIMATORS = 300

FIGURE_SIZE = (14, 6)

DPI = 300


def create_directories():
    folders = [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURE_DIR,
        FORECAST_DIR,
        METRIC_DIR,
        MODEL_DIR,
        REPORT_FIGURE_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
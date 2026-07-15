import json
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.config import FIGURE_DIR
from electricity_demand.config import FORECAST_DIR
from electricity_demand.config import METRIC_DIR
from electricity_demand.config import MODEL_DIR
from electricity_demand.config import INTERIM_DATA_DIR
from electricity_demand.config import PROCESSED_DATA_DIR
from electricity_demand.config import create_directories


create_directories()


def save_dataframe(dataframe, file_name, folder="processed"):
    if folder == "processed":
        save_path = PROCESSED_DATA_DIR / file_name
    else:
        save_path = INTERIM_DATA_DIR / file_name

    dataframe.to_csv(save_path, index=True)
    print(f"Saved dataframe to {save_path}")


def save_forecast(dataframe, file_name):
    save_path = FORECAST_DIR / file_name
    dataframe.to_csv(save_path, index=True)
    print(f"Saved forecast to {save_path}")


def save_metrics(metrics, file_name):
    save_path = METRIC_DIR / file_name

    if isinstance(metrics, pd.DataFrame):
        metrics.to_csv(save_path, index=False)

    elif isinstance(metrics, dict):
        pd.DataFrame([metrics]).to_csv(save_path, index=False)

    else:
        raise ValueError("Metrics must be a dictionary or DataFrame.")

    print(f"Saved metrics to {save_path}")


def save_json(data, file_name):
    save_path = METRIC_DIR / file_name

    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Saved json file to {save_path}")


def save_model(model, file_name):
    save_path = MODEL_DIR / file_name
    joblib.dump(model, save_path)
    print(f"Saved model to {save_path}")


def save_pickle(model, file_name):
    save_path = MODEL_DIR / file_name

    with open(save_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Saved pickle file to {save_path}")


def save_figure(file_name):
    save_path = FIGURE_DIR / file_name

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {save_path}")


def create_results_table(model_name, metrics):
    results = {"Model": model_name}

    results.update(metrics)

    return pd.DataFrame([results])


def merge_results(result_tables):
    return pd.concat(result_tables, ignore_index=True)


def save_results_table(results):
    save_path = METRIC_DIR / "all_model_results.csv"
    results.to_csv(save_path, index=False)
    print(f"Saved results table to {save_path}")


def print_dataset_information(dataframe):
    print()
    print("Dataset Shape")
    print(dataframe.shape)

    print()
    print("Column Names")
    print(dataframe.columns.tolist())

    print()
    print("Data Types")
    print(dataframe.dtypes)

    print()
    print("Missing Values")
    print(dataframe.isnull().sum())


def save_text(text, file_name):
    save_path = METRIC_DIR / file_name

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Saved text file to {save_path}")


def file_exists(file_path):
    return Path(file_path).exists()
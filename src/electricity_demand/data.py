import pandas as pd
import numpy as np

from electricity_demand.config import RAW_DATA_FILE
from electricity_demand.config import DATETIME_COLUMN
from electricity_demand.config import TARGET_COLUMN
from electricity_demand.config import COUNTRY_CODE
from electricity_demand.config import START_DATE

from electricity_demand.utils import save_dataframe
from electricity_demand.utils import print_dataset_information


def load_raw_data():
    print("Loading raw dataset")

    df = pd.read_csv(RAW_DATA_FILE)

    print(f"Dataset loaded successfully")
    print(f"Shape: {df.shape}")

    return df


def find_target_column(df):
    possible_columns = [
        f"{COUNTRY_CODE}_load_actual_entsoe_transparency",
        f"{COUNTRY_CODE}_load_actual_entsoe_power_statistics",
        TARGET_COLUMN,
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise ValueError("Electricity demand column was not found in the dataset.")


def clean_data(df):
    print("Cleaning dataset")

    demand_column = find_target_column(df)

    df = df[[DATETIME_COLUMN, demand_column]].copy()

    df.rename(
        columns={
            demand_column: "electricity_demand"
        },
        inplace=True,
    )

    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    df.dropna(subset=[DATETIME_COLUMN], inplace=True)

    df.sort_values(DATETIME_COLUMN, inplace=True)

    df.drop_duplicates(inplace=True)

    df.set_index(DATETIME_COLUMN, inplace=True)

    df = df[df.index >= START_DATE]

    df["electricity_demand"] = pd.to_numeric(
        df["electricity_demand"],
        errors="coerce",
    )

    df["electricity_demand"] = (
        df["electricity_demand"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    df = df.asfreq("H")

    return df


def dataset_summary(df):
    print_dataset_information(df)

    print()
    print("Summary Statistics")
    print(df.describe())

    print()
    print("Missing Values")
    print(df.isnull().sum())

    print()
    print("Duplicate Rows")
    print(df.duplicated().sum())


def create_hourly_dataset(df):
    hourly = df.copy()

    save_dataframe(
        hourly,
        "hourly_electricity_demand.csv",
        folder="processed",
    )

    return hourly


def create_daily_dataset(df):
    daily = df.resample("D").mean()

    save_dataframe(
        daily,
        "daily_electricity_demand.csv",
        folder="processed",
    )

    return daily


def create_weekly_dataset(df):
    weekly = df.resample("W").mean()

    save_dataframe(
        weekly,
        "weekly_electricity_demand.csv",
        folder="processed",
    )

    return weekly


def create_monthly_dataset(df):
    monthly = df.resample("M").mean()

    save_dataframe(
        monthly,
        "monthly_electricity_demand.csv",
        folder="processed",
    )

    return monthly


def create_yearly_dataset(df):
    yearly = df.resample("Y").mean()

    save_dataframe(
        yearly,
        "yearly_electricity_demand.csv",
        folder="processed",
    )

    return yearly


def prepare_data():
    raw_data = load_raw_data()

    cleaned_data = clean_data(raw_data)

    dataset_summary(cleaned_data)

    hourly_data = create_hourly_dataset(cleaned_data)

    daily_data = create_daily_dataset(cleaned_data)

    weekly_data = create_weekly_dataset(cleaned_data)

    monthly_data = create_monthly_dataset(cleaned_data)

    yearly_data = create_yearly_dataset(cleaned_data)

    save_dataframe(
        cleaned_data,
        "cleaned_electricity_demand.csv",
        folder="processed",
    )

    return {
        "hourly": hourly_data,
        "daily": daily_data,
        "weekly": weekly_data,
        "monthly": monthly_data,
        "yearly": yearly_data,
    }
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from electricity_demand.utils import save_dataframe
from electricity_demand.utils import save_model


class FeatureEngineering:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def create_features(self):
        print("Creating features")

        self.create_calendar_features()

        self.create_lag_features()

        self.create_rolling_features()

        self.create_cyclic_features()

        self.remove_missing_rows()

        save_dataframe(
            self.df,
            "feature_engineered_data.csv",
            folder="processed",
        )

        return self.df

    def create_calendar_features(self):
        self.df["year"] = self.df.index.year
        self.df["month"] = self.df.index.month
        self.df["day"] = self.df.index.day
        self.df["day_of_week"] = self.df.index.dayofweek
        self.df["day_of_year"] = self.df.index.dayofyear
        self.df["week_of_year"] = self.df.index.isocalendar().week.astype(int)
        self.df["quarter"] = self.df.index.quarter
        self.df["hour"] = self.df.index.hour
        self.df["is_weekend"] = (self.df.index.dayofweek >= 5).astype(int)

    def create_lag_features(self):
        self.df["lag_1"] = self.df["electricity_demand"].shift(1)
        self.df["lag_2"] = self.df["electricity_demand"].shift(2)
        self.df["lag_3"] = self.df["electricity_demand"].shift(3)
        self.df["lag_6"] = self.df["electricity_demand"].shift(6)
        self.df["lag_12"] = self.df["electricity_demand"].shift(12)
        self.df["lag_24"] = self.df["electricity_demand"].shift(24)
        self.df["lag_48"] = self.df["electricity_demand"].shift(48)
        self.df["lag_72"] = self.df["electricity_demand"].shift(72)
        self.df["lag_168"] = self.df["electricity_demand"].shift(168)

    def create_rolling_features(self):
        self.df["rolling_mean_6"] = (
            self.df["electricity_demand"]
            .rolling(6)
            .mean()
        )

        self.df["rolling_mean_12"] = (
            self.df["electricity_demand"]
            .rolling(12)
            .mean()
        )

        self.df["rolling_mean_24"] = (
            self.df["electricity_demand"]
            .rolling(24)
            .mean()
        )

        self.df["rolling_mean_168"] = (
            self.df["electricity_demand"]
            .rolling(168)
            .mean()
        )

        self.df["rolling_std_24"] = (
            self.df["electricity_demand"]
            .rolling(24)
            .std()
        )

        self.df["rolling_std_168"] = (
            self.df["electricity_demand"]
            .rolling(168)
            .std()
        )

        self.df["rolling_min_24"] = (
            self.df["electricity_demand"]
            .rolling(24)
            .min()
        )

        self.df["rolling_max_24"] = (
            self.df["electricity_demand"]
            .rolling(24)
            .max()
        )

    def create_cyclic_features(self):
        self.df["hour_sin"] = np.sin(
            2 * np.pi * self.df["hour"] / 24
        )

        self.df["hour_cos"] = np.cos(
            2 * np.pi * self.df["hour"] / 24
        )

        self.df["month_sin"] = np.sin(
            2 * np.pi * self.df["month"] / 12
        )

        self.df["month_cos"] = np.cos(
            2 * np.pi * self.df["month"] / 12
        )

        self.df["day_sin"] = np.sin(
            2 * np.pi * self.df["day_of_week"] / 7
        )

        self.df["day_cos"] = np.cos(
            2 * np.pi * self.df["day_of_week"] / 7
        )

    def remove_missing_rows(self):
        self.df.dropna(inplace=True)

    def prepare_machine_learning_data(self):
        target = "electricity_demand"

        X = self.df.drop(columns=[target])

        y = self.df[target]

        return X, y

    def standard_scale_features(self, X):
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        save_model(
            scaler,
            "standard_scaler.pkl",
        )

        return X_scaled

    def minmax_scale_features(self, values):
        scaler = MinMaxScaler()

        scaled_values = scaler.fit_transform(values)

        save_model(
            scaler,
            "minmax_scaler.pkl",
        )

        return scaled_values, scaler

    def create_lstm_sequences(self, values, sequence_length):
        X = []
        y = []

        for i in range(sequence_length, len(values)):
            X.append(values[i-sequence_length:i])
            y.append(values[i])

        X = np.array(X)
        y = np.array(y)

        return X, y
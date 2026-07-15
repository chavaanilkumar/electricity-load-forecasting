import numpy as np
import pandas as pd

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_forecast
from electricity_demand.utils import save_forecast


class BenchmarkModels:

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.evaluator = ModelEvaluation()

    def train_test_split(self):
        forecast_steps = 104

        train = self.df.iloc[:-forecast_steps]
        test = self.df.iloc[-forecast_steps:]

        return train, test

    def mean_forecast(self):
        train, test = self.train_test_split()

        forecast = np.repeat(
            train["electricity_demand"].mean(),
            len(test),
        )

        forecast = pd.Series(
            forecast,
            index=test.index,
            name="Forecast",
        )

        self.save_results(
            "Mean",
            train,
            test,
            forecast,
        )

        return self.evaluator.evaluate_model(
            "Mean",
            test["electricity_demand"],
            forecast,
        )

    def naive_forecast(self):
        train, test = self.train_test_split()

        forecast = np.repeat(
            train["electricity_demand"].iloc[-1],
            len(test),
        )

        forecast = pd.Series(
            forecast,
            index=test.index,
            name="Forecast",
        )

        self.save_results(
            "Naive",
            train,
            test,
            forecast,
        )

        return self.evaluator.evaluate_model(
            "Naive",
            test["electricity_demand"],
            forecast,
        )

    def seasonal_naive_forecast(self):
        train, test = self.train_test_split()

        seasonal_period = 52

        history = train["electricity_demand"].tolist()

        predictions = []

        for i in range(len(test)):
            value = history[-seasonal_period]
            predictions.append(value)
            history.append(value)

        forecast = pd.Series(
            predictions,
            index=test.index,
            name="Forecast",
        )

        self.save_results(
            "Seasonal_Naive",
            train,
            test,
            forecast,
        )

        return self.evaluator.evaluate_model(
            "Seasonal Naive",
            test["electricity_demand"],
            forecast,
        )

    def drift_forecast(self):
        train, test = self.train_test_split()

        first = train["electricity_demand"].iloc[0]
        last = train["electricity_demand"].iloc[-1]

        slope = (last - first) / (len(train) - 1)

        forecast = []

        for step in range(1, len(test) + 1):
            forecast.append(last + slope * step)

        forecast = pd.Series(
            forecast,
            index=test.index,
            name="Forecast",
        )

        self.save_results(
            "Drift",
            train,
            test,
            forecast,
        )

        return self.evaluator.evaluate_model(
            "Drift",
            test["electricity_demand"],
            forecast,
        )

    def save_results(
        self,
        model_name,
        train,
        test,
        forecast,
    ):
        forecast_df = pd.DataFrame(
            {
                "Actual": test["electricity_demand"],
                "Forecast": forecast,
            }
        )

        save_forecast(
            forecast_df,
            f"{model_name.lower()}_forecast.csv",
        )

        plot_forecast(
            train["electricity_demand"],
            test["electricity_demand"],
            forecast,
            f"{model_name} Forecast",
            f"{model_name.lower()}_forecast.png",
        )

    def run_all_models(self):
        results = []

        results.append(self.mean_forecast())
        results.append(self.naive_forecast())
        results.append(self.seasonal_naive_forecast())
        results.append(self.drift_forecast())

        final_results = self.evaluator.evaluate_multiple_models(
            results
        )

        return final_results
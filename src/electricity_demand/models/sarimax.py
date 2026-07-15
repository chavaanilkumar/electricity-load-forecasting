import warnings

import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_forecast_with_confidence
from electricity_demand.plotting import plot_residual_distribution
from electricity_demand.plotting import plot_residuals
from electricity_demand.utils import save_forecast
from electricity_demand.utils import save_model
from electricity_demand.utils import save_text

warnings.filterwarnings("ignore")


class SARIMAModel:

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.evaluator = ModelEvaluation()

    def train_test_split(self):
        forecast_steps = 104

        train = self.df.iloc[:-forecast_steps]

        test = self.df.iloc[-forecast_steps:]

        return train, test

    def train_model(self):
        train, test = self.train_test_split()

        order = (2, 1, 2)

        seasonal_order = (1, 1, 1, 52)

        save_text(
            f"SARIMA Order : {order}\nSeasonal Order : {seasonal_order}",
            "best_sarima_model.txt",
        )

        print("Training SARIMA model")

        model = SARIMAX(
            train["electricity_demand"],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(
            disp=False,
            maxiter=100,
        )

        save_model(
            fitted_model,
            "sarima_model.pkl",
        )

        forecast = fitted_model.get_forecast(
            steps=len(test)
        )

        forecast_values = forecast.predicted_mean

        confidence = forecast.conf_int()

        lower = confidence.iloc[:, 0]

        upper = confidence.iloc[:, 1]

        forecast_df = pd.DataFrame(
            {
                "Actual": test["electricity_demand"],
                "Forecast": forecast_values,
                "Lower": lower,
                "Upper": upper,
            },
            index=test.index,
        )

        save_forecast(
            forecast_df,
            "sarima_forecast.csv",
        )

        plot_forecast_with_confidence(
            train["electricity_demand"],
            test["electricity_demand"],
            forecast_values,
            lower,
            upper,
            "SARIMA Forecast",
            "sarima_forecast.png",
        )

        residuals = fitted_model.resid

        plot_residuals(
            residuals,
            "SARIMA Residuals",
            "sarima_residuals.png",
        )

        plot_residual_distribution(
            residuals,
            "SARIMA Residual Distribution",
            "sarima_residual_distribution.png",
        )

        results = self.evaluator.evaluate_model(
            "SARIMA",
            test["electricity_demand"],
            forecast_values,
        )

        return results

    def train_sarimax(self, exogenous_data):
        train, test = self.train_test_split()

        train_x = exogenous_data.iloc[:-104]

        test_x = exogenous_data.iloc[-104:]

        order = (2, 1, 2)

        seasonal_order = (1, 1, 1, 52)

        print("Training SARIMAX model")

        model = SARIMAX(
            train["electricity_demand"],
            exog=train_x,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(
            disp=False,
            maxiter=100,
        )

        save_model(
            fitted_model,
            "sarimax_model.pkl",
        )

        forecast = fitted_model.get_forecast(
            steps=len(test),
            exog=test_x,
        )

        forecast_values = forecast.predicted_mean

        confidence = forecast.conf_int()

        lower = confidence.iloc[:, 0]

        upper = confidence.iloc[:, 1]

        forecast_df = pd.DataFrame(
            {
                "Actual": test["electricity_demand"],
                "Forecast": forecast_values,
                "Lower": lower,
                "Upper": upper,
            },
            index=test.index,
        )

        save_forecast(
            forecast_df,
            "sarimax_forecast.csv",
        )

        plot_forecast_with_confidence(
            train["electricity_demand"],
            test["electricity_demand"],
            forecast_values,
            lower,
            upper,
            "SARIMAX Forecast",
            "sarimax_forecast.png",
        )

        results = self.evaluator.evaluate_model(
            "SARIMAX",
            test["electricity_demand"],
            forecast_values,
        )

        return results
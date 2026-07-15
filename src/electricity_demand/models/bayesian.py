import numpy as np
import pandas as pd

from sklearn.linear_model import BayesianRidge

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_forecast
from electricity_demand.utils import save_forecast
from electricity_demand.utils import save_model


class BayesianRegressionModel:

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.evaluator = ModelEvaluation()

    def prepare_data(self):
        target = "electricity_demand"

        X = self.df.drop(columns=[target])
        y = self.df[target]

        forecast_steps = 104

        X_train = X.iloc[:-forecast_steps]
        X_test = X.iloc[-forecast_steps:]

        y_train = y.iloc[:-forecast_steps]
        y_test = y.iloc[-forecast_steps:]

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        model = BayesianRidge()

        model.fit(X_train, y_train)

        save_model(
            model,
            "bayesian_ridge.pkl",
        )

        predictions, prediction_std = model.predict(
            X_test,
            return_std=True,
        )

        forecast = pd.DataFrame(
            {
                "Actual": y_test,
                "Forecast": predictions,
                "Prediction_Std": prediction_std,
            },
            index=y_test.index,
        )

        save_forecast(
            forecast,
            "bayesian_forecast.csv",
        )

        plot_forecast(
            y_train,
            y_test,
            pd.Series(predictions, index=y_test.index),
            "Bayesian Regression Forecast",
            "bayesian_forecast.png",
        )

        results = self.evaluator.evaluate_model(
            "Bayesian Ridge",
            y_test,
            predictions,
        )

        return results

    def predict(self, X):
        return self.model.predict(X)
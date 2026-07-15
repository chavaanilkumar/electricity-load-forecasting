import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_forecast
from electricity_demand.plotting import plot_feature_importance
from electricity_demand.utils import save_forecast
from electricity_demand.utils import save_model


class FeatureBasedModels:

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

    def random_forest(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        save_model(
            model,
            "random_forest.pkl",
        )

        predictions = model.predict(X_test)

        forecast = pd.DataFrame(
            {
                "Actual": y_test,
                "Forecast": predictions,
            },
            index=y_test.index,
        )

        save_forecast(
            forecast,
            "random_forest_forecast.csv",
        )

        plot_forecast(
            y_train,
            y_test,
            pd.Series(predictions, index=y_test.index),
            "Random Forest Forecast",
            "random_forest_forecast.png",
        )

        plot_feature_importance(
            model,
            X_train.columns,
            "Random Forest Feature Importance",
            "random_forest_feature_importance.png",
        )

        return self.evaluator.evaluate_model(
            "Random Forest",
            y_test,
            predictions,
        )

    def gradient_boosting(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

        model.fit(X_train, y_train)

        save_model(
            model,
            "gradient_boosting.pkl",
        )

        predictions = model.predict(X_test)

        forecast = pd.DataFrame(
            {
                "Actual": y_test,
                "Forecast": predictions,
            },
            index=y_test.index,
        )

        save_forecast(
            forecast,
            "gradient_boosting_forecast.csv",
        )

        plot_forecast(
            y_train,
            y_test,
            pd.Series(predictions, index=y_test.index),
            "Gradient Boosting Forecast",
            "gradient_boosting_forecast.png",
        )

        plot_feature_importance(
            model,
            X_train.columns,
            "Gradient Boosting Feature Importance",
            "gradient_boosting_feature_importance.png",
        )

        return self.evaluator.evaluate_model(
            "Gradient Boosting",
            y_test,
            predictions,
        )

    def run_all_models(self):
        results = []

        results.append(self.random_forest())

        results.append(self.gradient_boosting())

        final_results = self.evaluator.evaluate_multiple_models(
            results
        )

        return final_results
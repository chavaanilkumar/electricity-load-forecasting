import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from electricity_demand.utils import save_metrics
from electricity_demand.utils import create_results_table


class ModelEvaluation:

    def __init__(self):
        pass

    def calculate_metrics(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        r2 = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)

        bias = np.mean(predicted - actual)

        metrics = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 4),
            "MSE": round(mse, 4),
            "R2": round(r2, 4),
            "Bias": round(bias, 4),
        }

        return metrics

    def evaluate_model(
        self,
        model_name,
        actual,
        predicted,
    ):
        metrics = self.calculate_metrics(
            actual,
            predicted,
        )

        save_metrics(
            metrics,
            f"{model_name.lower().replace(' ', '_')}_metrics.csv",
        )

        results = create_results_table(
            model_name,
            metrics,
        )

        return results

    def evaluate_multiple_models(
        self,
        results_list,
    ):
        final_results = pd.concat(
            results_list,
            ignore_index=True,
        )

        final_results = final_results.sort_values(
            by="RMSE",
            ascending=True,
        )

        save_metrics(
            final_results,
            "all_model_results.csv",
        )

        return final_results

    def print_metrics(self, metrics):
        print()

        for key, value in metrics.items():
            print(f"{key}: {value}")

    def residuals(
        self,
        actual,
        predicted,
    ):
        return np.array(actual) - np.array(predicted)

    def residual_summary(
        self,
        residuals,
    ):
        summary = pd.DataFrame(
            {
                "Statistic": [
                    "Mean",
                    "Median",
                    "Standard Deviation",
                    "Minimum",
                    "Maximum",
                ],
                "Value": [
                    residuals.mean(),
                    np.median(residuals),
                    residuals.std(),
                    residuals.min(),
                    residuals.max(),
                ],
            }
        )

        save_metrics(
            summary,
            "residual_summary.csv",
        )

        return summary

    def prediction_dataframe(
        self,
        index,
        actual,
        predicted,
    ):
        predictions = pd.DataFrame(
            {
                "Date": index,
                "Actual": actual,
                "Predicted": predicted,
                "Residual": np.array(actual) - np.array(predicted),
            }
        )

        predictions.to_csv(
            "outputs/forecasts/predictions.csv",
            index=False,
        )

        return predictions

    def compare_models(
        self,
        results_dataframe,
    ):
        print()

        print("Model Comparison")

        print(results_dataframe.sort_values("RMSE"))

        return results_dataframe.sort_values("RMSE")

    def best_model(
        self,
        results_dataframe,
    ):
        best = results_dataframe.sort_values(
            "RMSE"
        ).iloc[0]

        print()

        print("Best Model")

        print(best)

        return best
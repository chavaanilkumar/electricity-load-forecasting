import pandas as pd

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_model_comparison


def main():
    results_file = "outputs/metrics/all_model_results.csv"

    results = pd.read_csv(results_file)

    evaluator = ModelEvaluation()

    print()
    print("All Model Results")
    print(results)

    print()

    best_model = evaluator.best_model(results)

    print()
    print("Best Performing Model")
    print(best_model)

    plot_model_comparison(
        results,
        "model_comparison_rmse.png",
    )


if __name__ == "__main__":
    main()
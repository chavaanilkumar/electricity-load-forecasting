import numpy as np

from electricity_demand.evaluation import ModelEvaluation


def test_model_evaluation():
    actual = np.array([100, 120, 130, 150, 170, 180])

    predicted = np.array([102, 118, 132, 148, 171, 179])

    evaluator = ModelEvaluation()

    metrics = evaluator.calculate_metrics(
        actual,
        predicted,
    )

    assert metrics["RMSE"] >= 0
    assert metrics["MAE"] >= 0
    assert metrics["MAPE"] >= 0

    print()
    print("Evaluation metrics")
    print(metrics)
    print()

    print("Evaluation test passed")


if __name__ == "__main__":
    test_model_evaluation()
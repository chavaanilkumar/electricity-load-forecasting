from electricity_demand.config import create_directories

from electricity_demand.data import prepare_data

from electricity_demand.eda import ElectricityEDA

from electricity_demand.features import FeatureEngineering

from electricity_demand.models.benchmarks import BenchmarkModels

from electricity_demand.models.sarimax import SARIMAModel

from electricity_demand.models.feature_models import FeatureBasedModels

from electricity_demand.models.bayesian import BayesianRegressionModel

from electricity_demand.models.neural import LSTMForecastModel

from electricity_demand.evaluation import ModelEvaluation

from electricity_demand.utils import save_results_table


class ElectricityDemandPipeline:

    def __init__(self):
        create_directories()

        self.results = []

    def run(self):
        print("Starting electricity demand forecasting pipeline")

        datasets = prepare_data()

        hourly_data = datasets["hourly"]

        weekly_data = datasets["weekly"]

        print("Running exploratory data analysis")

        eda = ElectricityEDA(hourly_data)

        eda.run_all()

        print("Creating machine learning features")

        feature_engineering = FeatureEngineering(hourly_data)

        feature_data = feature_engineering.create_features()

        print("Running benchmark models")

        benchmark_models = BenchmarkModels(weekly_data)

        benchmark_results = benchmark_models.run_all_models()

        self.results.append(benchmark_results)

        print("Running SARIMA model")

        sarima = SARIMAModel(weekly_data)

        sarima_results = sarima.train_model()

        self.results.append(sarima_results)

        print("Running feature based models")

        feature_models = FeatureBasedModels(feature_data)

        rf_results = feature_models.random_forest()

        gb_results = feature_models.gradient_boosting()

        self.results.append(rf_results)

        self.results.append(gb_results)

        print("Running Bayesian regression")

        bayesian = BayesianRegressionModel(feature_data)

        bayesian_results = bayesian.train_model()

        self.results.append(bayesian_results)

        print("Running LSTM")

        lstm = LSTMForecastModel(hourly_data)

        lstm_results = lstm.train_model()

        self.results.append(lstm_results)

        print("Combining evaluation results")

        evaluator = ModelEvaluation()

        final_results = evaluator.evaluate_multiple_models(self.results)

        save_results_table(final_results)

        print()

        print(final_results)

        print()

        print("Pipeline completed successfully")

        return final_results
from electricity_demand.data import prepare_data
from electricity_demand.models.benchmarks import BenchmarkModels


def test_benchmark_models():
    datasets = prepare_data()

    weekly_data = datasets["weekly"]

    benchmark = BenchmarkModels(weekly_data)

    results = benchmark.run_all_models()

    assert results is not None

    assert len(results) == 4

    print()
    print("Benchmark Model Results")
    print(results)
    print()

    print("Benchmark model test passed")


if __name__ == "__main__":
    test_benchmark_models()
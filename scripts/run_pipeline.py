from electricity_demand.pipeline import ElectricityDemandPipeline


def main():
    pipeline = ElectricityDemandPipeline()

    results = pipeline.run()

    print()
    print("Final Model Comparison")
    print(results)


if __name__ == "__main__":
    main()
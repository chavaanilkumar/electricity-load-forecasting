from electricity_demand.data import prepare_data
from electricity_demand.features import FeatureEngineering


def main():
    datasets = prepare_data()

    hourly_data = datasets["hourly"]

    feature_engineering = FeatureEngineering(hourly_data)

    feature_data = feature_engineering.create_features()

    print()
    print("Feature engineering completed successfully")
    print()
    print(feature_data.head())
    print()
    print(f"Final dataset shape: {feature_data.shape}")


if __name__ == "__main__":
    main()
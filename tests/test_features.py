from electricity_demand.data import prepare_data
from electricity_demand.features import FeatureEngineering


def test_feature_creation():
    datasets = prepare_data()

    hourly_data = datasets["hourly"]

    feature_engineering = FeatureEngineering(hourly_data)

    feature_data = feature_engineering.create_features()

    assert feature_data is not None

    assert "electricity_demand" in feature_data.columns

    assert len(feature_data.columns) > 10

    assert feature_data.isnull().sum().sum() == 0

    print("Feature engineering test passed")


if __name__ == "__main__":
    test_feature_creation()
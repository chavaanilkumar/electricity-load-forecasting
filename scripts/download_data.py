from pathlib import Path

from electricity_demand.config import RAW_DATA_FILE


def main():
    print()

    if Path(RAW_DATA_FILE).exists():
        print("Raw dataset found")
        print(f"Location : {RAW_DATA_FILE}")
    else:
        print("Raw dataset was not found")
        print("Copy the dataset into the folder below")
        print("data/raw/time_series_60min_singleindex.csv")

    print()


if __name__ == "__main__":
    main()
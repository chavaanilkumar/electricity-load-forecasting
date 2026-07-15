import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import probplot

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from electricity_demand.utils import save_figure
from electricity_demand.utils import save_text


warnings.filterwarnings("ignore")


class ElectricityEDA:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def run_all(self):
        print("Running exploratory data analysis")

        self.basic_information()
        self.summary_statistics()
        self.missing_values()
        self.time_series_plot()
        self.histogram()
        self.boxplot()
        self.monthly_boxplot()
        self.yearly_boxplot()
        self.hourly_pattern()
        self.weekday_pattern()
        self.monthly_pattern()
        self.yearly_pattern()
        self.correlation_heatmap()
        self.rolling_statistics()
        self.seasonal_decomposition()
        self.adf_test()
        self.kpss_test()
        self.acf_plot()
        self.pacf_plot()
        self.qq_plot()

        print("EDA completed")

    def basic_information(self):
        text = []

        text.append(f"Shape : {self.df.shape}")
        text.append("")
        text.append(str(self.df.dtypes))
        text.append("")
        text.append(str(self.df.head()))

        save_text("\n".join(text), "dataset_information.txt")

    def summary_statistics(self):
        summary = self.df.describe()

        summary.to_csv("outputs/metrics/summary_statistics.csv")

    def missing_values(self):
        missing = self.df.isnull().sum()

        missing.to_csv("outputs/metrics/missing_values.csv")

        plt.figure(figsize=(10, 5))

        sns.heatmap(
            self.df.isnull(),
            cbar=False,
            yticklabels=False,
        )

        plt.title("Missing Values")

        save_figure("missing_values.png")

    def time_series_plot(self):
        plt.figure(figsize=(16, 6))

        plt.plot(
            self.df.index,
            self.df["electricity_demand"],
        )

        plt.title("Electricity Demand")

        plt.xlabel("Date")

        plt.ylabel("Demand")

        save_figure("time_series.png")

    def histogram(self):
        plt.figure(figsize=(10, 6))

        sns.histplot(
            self.df["electricity_demand"],
            kde=True,
        )

        plt.title("Demand Distribution")

        save_figure("distribution.png")

    def boxplot(self):
        plt.figure(figsize=(8, 5))

        sns.boxplot(
            y=self.df["electricity_demand"],
        )

        plt.title("Demand Boxplot")

        save_figure("boxplot.png")

    def monthly_boxplot(self):
        temp = self.df.copy()

        temp["Month"] = temp.index.month_name()

        order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        plt.figure(figsize=(14, 6))

        sns.boxplot(
            data=temp,
            x="Month",
            y="electricity_demand",
            order=order,
        )

        plt.xticks(rotation=45)

        plt.title("Monthly Demand")

        save_figure("monthly_boxplot.png")

    def yearly_boxplot(self):
        temp = self.df.copy()

        temp["Year"] = temp.index.year

        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=temp,
            x="Year",
            y="electricity_demand",
        )

        plt.title("Yearly Demand")

        save_figure("yearly_boxplot.png")

    def hourly_pattern(self):
        temp = self.df.copy()

        temp["Hour"] = temp.index.hour

        hourly = temp.groupby("Hour")["electricity_demand"].mean()

        plt.figure(figsize=(10, 5))

        plt.plot(hourly)

        plt.title("Average Hourly Demand")

        save_figure("hourly_pattern.png")

    def weekday_pattern(self):
        temp = self.df.copy()

        temp["Weekday"] = temp.index.day_name()

        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        weekday = (
            temp.groupby("Weekday")["electricity_demand"]
            .mean()
            .reindex(order)
        )

        plt.figure(figsize=(10, 5))

        plt.plot(weekday)

        plt.title("Average Weekday Demand")

        save_figure("weekday_pattern.png")

    def monthly_pattern(self):
        monthly = self.df.resample("M").mean()

        plt.figure(figsize=(16, 5))

        plt.plot(monthly.index, monthly["electricity_demand"])

        plt.title("Monthly Demand")

        save_figure("monthly_trend.png")

    def yearly_pattern(self):
        yearly = self.df.resample("Y").mean()

        plt.figure(figsize=(10, 5))

        plt.plot(yearly.index, yearly["electricity_demand"])

        plt.title("Yearly Demand")

        save_figure("yearly_trend.png")

    def correlation_heatmap(self):
        temp = self.df.copy()

        temp["Hour"] = temp.index.hour
        temp["Day"] = temp.index.day
        temp["Month"] = temp.index.month
        temp["Year"] = temp.index.year

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            temp.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
        )

        plt.title("Correlation Heatmap")

        save_figure("correlation_heatmap.png")

    def rolling_statistics(self):
        rolling_mean = self.df["electricity_demand"].rolling(24 * 30).mean()

        rolling_std = self.df["electricity_demand"].rolling(24 * 30).std()

        plt.figure(figsize=(16, 6))

        plt.plot(
            self.df.index,
            self.df["electricity_demand"],
            label="Original",
        )

        plt.plot(
            rolling_mean,
            label="Rolling Mean",
        )

        plt.plot(
            rolling_std,
            label="Rolling Std",
        )

        plt.legend()

        plt.title("Rolling Statistics")

        save_figure("rolling_statistics.png")

    def seasonal_decomposition(self):
        weekly = self.df.resample("D").mean()

        result = seasonal_decompose(
            weekly["electricity_demand"],
            model="additive",
            period=365,
        )

        fig = result.plot()

        fig.set_size_inches(14, 10)

        save_figure("seasonal_decomposition.png")

    def adf_test(self):
        result = adfuller(
            self.df["electricity_demand"].dropna()
        )

        text = []

        text.append(f"ADF Statistic : {result[0]}")
        text.append(f"P Value : {result[1]}")

        save_text("\n".join(text), "adf_test.txt")

    def kpss_test(self):
        result = kpss(
            self.df["electricity_demand"].dropna(),
            regression="c",
        )

        text = []

        text.append(f"KPSS Statistic : {result[0]}")
        text.append(f"P Value : {result[1]}")

        save_text("\n".join(text), "kpss_test.txt")

    def acf_plot(self):
        plt.figure(figsize=(12, 5))

        plot_acf(
            self.df["electricity_demand"].dropna(),
            lags=60,
        )

        save_figure("acf_plot.png")

    def pacf_plot(self):
        plt.figure(figsize=(12, 5))

        plot_pacf(
            self.df["electricity_demand"].dropna(),
            lags=60,
            method="ywm",
        )

        save_figure("pacf_plot.png")

    def qq_plot(self):
        plt.figure(figsize=(8, 8))

        probplot(
            self.df["electricity_demand"],
            dist="norm",
            plot=plt,
        )

        plt.title("QQ Plot")

        save_figure("qq_plot.png")
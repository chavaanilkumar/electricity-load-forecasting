import matplotlib.pyplot as plt
import seaborn as sns

from electricity_demand.utils import save_figure


plt.style.use("default")

sns.set_theme(style="whitegrid")


def plot_actual_series(dataframe, title, file_name):
    plt.figure(figsize=(16, 6))

    plt.plot(
        dataframe.index,
        dataframe["electricity_demand"],
        linewidth=1.5,
        label="Actual",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.legend()

    save_figure(file_name)


def plot_train_test(train, test, title, file_name):
    plt.figure(figsize=(16, 6))

    plt.plot(
        train.index,
        train,
        label="Training",
    )

    plt.plot(
        test.index,
        test,
        label="Testing",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.legend()

    save_figure(file_name)


def plot_forecast(
    train,
    test,
    forecast,
    title,
    file_name,
):
    plt.figure(figsize=(16, 6))

    plt.plot(
        train.index,
        train,
        label="Training",
    )

    plt.plot(
        test.index,
        test,
        label="Actual",
    )

    plt.plot(
        forecast.index,
        forecast,
        label="Forecast",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.legend()

    save_figure(file_name)


def plot_forecast_with_confidence(
    train,
    test,
    forecast,
    lower,
    upper,
    title,
    file_name,
):
    plt.figure(figsize=(16, 6))

    plt.plot(
        train.index,
        train,
        label="Training",
    )

    plt.plot(
        test.index,
        test,
        label="Actual",
    )

    plt.plot(
        forecast.index,
        forecast,
        label="Forecast",
    )

    plt.fill_between(
        forecast.index,
        lower,
        upper,
        alpha=0.25,
        label="95% Confidence Interval",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.legend()

    save_figure(file_name)


def plot_residuals(residuals, title, file_name):
    plt.figure(figsize=(14, 5))

    plt.plot(residuals)

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    plt.title(title)
    plt.xlabel("Observation")
    plt.ylabel("Residual")

    save_figure(file_name)


def plot_residual_distribution(residuals, title, file_name):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        residuals,
        kde=True,
    )

    plt.title(title)

    save_figure(file_name)


def plot_feature_importance(model, feature_names, title, file_name):
    importance = model.feature_importances_

    order = importance.argsort()[::-1]

    plt.figure(figsize=(12, 8))

    plt.barh(
        range(len(order)),
        importance[order],
    )

    plt.yticks(
        range(len(order)),
        [feature_names[i] for i in order],
    )

    plt.title(title)

    save_figure(file_name)


def plot_lstm_loss(history, file_name):
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title("LSTM Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    save_figure(file_name)


def plot_model_comparison(results, file_name):
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results,
        x="Model",
        y="RMSE",
    )

    plt.xticks(rotation=30)

    plt.title("Model Comparison Using RMSE")

    save_figure(file_name)


def plot_correlation_matrix(dataframe, file_name):
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        dataframe.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False,
    )

    plt.title("Feature Correlation Matrix")

    save_figure(file_name)


def plot_predictions(actual, predicted, title, file_name):
    plt.figure(figsize=(16, 6))

    plt.plot(
        actual.index,
        actual,
        label="Actual",
    )

    plt.plot(
        predicted.index,
        predicted,
        label="Predicted",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand")
    plt.legend()

    save_figure(file_name)
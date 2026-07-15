import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from electricity_demand.evaluation import ModelEvaluation
from electricity_demand.plotting import plot_forecast
from electricity_demand.plotting import plot_lstm_loss
from electricity_demand.utils import save_forecast
from electricity_demand.utils import save_model


class LSTMForecastModel:

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.evaluator = ModelEvaluation()

    def prepare_data(self):
        values = self.df["electricity_demand"].values.reshape(-1, 1)

        scaler = MinMaxScaler()

        scaled_values = scaler.fit_transform(values)

        save_model(
            scaler,
            "lstm_scaler.pkl",
        )

        sequence_length = 24

        X = []
        y = []

        for i in range(sequence_length, len(scaled_values)):
            X.append(scaled_values[i-sequence_length:i])
            y.append(scaled_values[i])

        X = np.array(X)
        y = np.array(y)

        forecast_steps = 24 * 365 * 2

        X_train = X[:-forecast_steps]
        X_test = X[-forecast_steps:]

        y_train = y[:-forecast_steps]
        y_test = y[-forecast_steps:]

        return X_train, X_test, y_train, y_test, scaler

    def build_model(self, input_shape):
        model = Sequential()

        model.add(
            LSTM(
                128,
                return_sequences=True,
                input_shape=input_shape,
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            LSTM(
                64,
                return_sequences=True,
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            LSTM(
                32,
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(16, activation="relu")
        )

        model.add(
            Dense(1)
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train_model(self):
        X_train, X_test, y_train, y_test, scaler = self.prepare_data()

        model = self.build_model(
            (X_train.shape[1], X_train.shape[2])
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
        )

        checkpoint = ModelCheckpoint(
            "outputs/model_objects/best_lstm_model.keras",
            monitor="val_loss",
            save_best_only=True,
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                early_stopping,
                reduce_lr,
                checkpoint,
            ],
            verbose=1,
        )

        save_model(
            model,
            "lstm_model.pkl",
        )

        plot_lstm_loss(
            history,
            "lstm_training_loss.png",
        )

        predictions = model.predict(
            X_test,
            verbose=0,
        )

        predictions = scaler.inverse_transform(
            predictions
        )

        y_test = scaler.inverse_transform(
            y_test
        )

        forecast = pd.DataFrame(
            {
                "Actual": y_test.flatten(),
                "Forecast": predictions.flatten(),
            },
            index=self.df.index[-len(predictions):],
        )

        save_forecast(
            forecast,
            "lstm_forecast.csv",
        )

        plot_forecast(
            pd.Series(
                self.df["electricity_demand"].iloc[:-len(predictions)],
                index=self.df.index[:-len(predictions)],
            ),
            pd.Series(
                y_test.flatten(),
                index=forecast.index,
            ),
            pd.Series(
                predictions.flatten(),
                index=forecast.index,
            ),
            "LSTM Forecast",
            "lstm_forecast.png",
        )

        results = self.evaluator.evaluate_model(
            "LSTM",
            y_test.flatten(),
            predictions.flatten(),
        )

        return results

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_dataset() -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv("weekly_adjusted_AAPL.csv")
    data = data[['timestamp', 'adjusted close']]
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.sort_values('timestamp', inplace=True)

    # Prepare the data
    data['adjusted close'] = data['adjusted close'].astype(float)
    series = data['adjusted close'].values

    return series


def main() -> None:
    # Prepare data for ARIMA
    series = load_dataset()
    # Split the dataset
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # ARIMA model
    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # Evaluate the model
    rmse = sqrt(mean_squared_error(test, predictions))
    print(f"Test RMSE: {rmse:.3f}")

    # Plot the results
    plt.plot(test, label="Actual")
    plt.plot(predictions, label="Predicted", color="red")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
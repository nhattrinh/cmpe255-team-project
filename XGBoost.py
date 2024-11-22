import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import asarray
from pandas import DataFrame, concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


DATA_FILENAME = "weekly_adjusted_SPY.csv"
STOCK_SYMBOL = "SPY"


def load_dataset() -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv(f"data/{DATA_FILENAME}")
    data = data[['timestamp', 'adjusted close', 'volume']]
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.sort_values('timestamp', inplace=True)

    # Prepare the data
    data['adjusted close'] = data['adjusted close'].astype(float)
    data['volume'] = data['volume'].astype(float)
    series = data[['adjusted close', 'volume']].values

    return series


# Transform a time series dataset into supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# Split dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


def xgboost_forecast(train, testX):
    # XGBoost model
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict(asarray([testX]))
    return yhat[0]


def walk_forward_validation(data, n_test):
    # Walk-forward validation
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, test_y = test[i, :-1], test[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
    return test[:, -1], predictions


def main() -> None:
    # Prepare data
    series = load_dataset()
    n_in = 6
    supervised = series_to_supervised(series, n_in=n_in)
    n_test = 50

    # Evaluate
    y, yhat = walk_forward_validation(supervised, n_test)
    plt.title(
        f'Stock Forecast vs Actual for {STOCK_SYMBOL} - '
        + pd.Timestamp.now().strftime("%Y-%m-%d")
    )
    plt.plot(y, label='Expected')
    plt.plot(yhat, label='Predicted')
    plt.legend()
    # If FigureCanvasAgg is interactive show plot
    if matplotlib.is_interactive():
        plt.show()
    else:
        plt.savefig(f'output/XGBoost_{STOCK_SYMBOL}_forecast.png')


if __name__ == '__main__':
    main()

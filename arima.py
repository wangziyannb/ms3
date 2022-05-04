# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/3/22 1:09 AM
# @Project : 
# @Author  : Ziyan Wang
# @Email   : ziyan.wang@stonybrook.edu
# @File    : arima.py
# @Software: PyCharm
import concurrent.futures

from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
# support multi thread processing
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.simplefilter("ignore")


def data_info(data):
    print(data.head(10))
    pyplot.subplot(1, 2, 1)
    data.plot()
    # generate correlation plot
    pyplot.subplot(1, 2, 2)
    autocorrelation_plot(data)
    # There is a positive correlation with the first 40 - 45 lags. And that is significant
    # for the first 7 lags.
    pyplot.show()
    return 7


def gen_model(data, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    return model


def model_info(data, p, d, q):
    model = gen_model(data, p, d, q)
    model_fit = model.fit()
    print(model_fit.summary())
    res = DataFrame(model_fit.resid)
    res.plot()
    pyplot.show()
    res.plot(kind='kde')
    pyplot.show()
    print(res.describe())


def test_params(train, test, p, d, q):
    history = [x for x in train]
    predictions = list()
    test_list = [x for x in test]
    for t in range(len(test)):
        model = gen_model(history, p, d, q)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    rmse = sqrt(mean_squared_error(test, predictions))
    # print('Test RMSE: %.3f' % rmse)
    return rmse, predictions, test_list, (p, d, q)


def grid_search_for_best_params(X, size):
    train, test = X[0:size], X[size:len(X)]
    p_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 24]
    d_value = range(0, 3)
    q_value = range(0, 3)
    m = {}
    test_list = None
    best_rmse, best_cfg = float("inf"), None
    pool = ProcessPoolExecutor(max_workers=20)
    thread_list = []
    for p in p_value:
        for d in d_value:
            for q in q_value:
                try:
                    thread_list.append(pool.submit(test_params, train, test, p, d, q))
                    # rmse, predictions, test_list = test_params(train, test, p, d, q)
                except:
                    continue
    print(thread_list)
    concurrent.futures.wait(thread_list, return_when=concurrent.futures.FIRST_COMPLETED)
    for f in as_completed(thread_list):
        try:
            rmse, predictions, test_list, (p, d, q) = f.result()
            m[(p, d, q)] = {"rmse": rmse, "pred": predictions}
            print("job for searching (%d, %d, %d) finished" % (p, d, q))
            if best_rmse > rmse:
                best_rmse = rmse
                best_cfg = (p, d, q)
        except:
            continue

    print("tried params:")
    print(m)
    print("best params:")
    print("best cfg:", best_cfg)
    print("best rmse:", best_rmse)
    print("best prediction history:")
    pred = m[best_cfg]["pred"]
    for i in range(len(pred)):
        print('predicted=%f, expected=%f' % (pred[i], test_list[i]))
    pyplot.plot(test_list)
    pyplot.plot(pred, color='red')
    pyplot.show()


if __name__ == '__main__':
    # load dataset
    data = read_csv('monthly-robberies.csv', parse_dates=[0], index_col=0).squeeze("columns")
    data.index = data.index.to_period('M')
    # p = data_info(data)
    # model_info(data, p, 1, 0)
    X = data.values
    size = int(len(X) * 0.66)
    grid_search_for_best_params(X, size)

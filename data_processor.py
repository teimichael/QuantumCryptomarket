import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./data/BTC_USD_2013-10-01_2019-12-15-CoinDesk.csv')
data_date = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data_cp = data['Closing Price (USD)']


def get_log_return(precision=3):
    log_return = np.diff(np.log(data_cp))
    precision = precision
    log_return = pd.DataFrame(np.around(log_return, decimals=precision))
    log_return_frequency = log_return[0].value_counts()
    log_return_frequency.sort_index(inplace=True)
    log_return_pdf = log_return_frequency / sum(log_return_frequency)
    interval_length = 10 ** (-precision)
    return log_return_pdf / sum(log_return_pdf * interval_length)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class DataProcessor:

    def __init__(self, data):
        self.data_date = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        self.data_cp = data['Closing Price (USD)']
        self.data_size = len(self.data_cp)

    def __get_log_return(self, precision=3):
        log_return = np.diff(np.log(self.data_cp))
        precision = precision
        log_return = pd.DataFrame(np.around(log_return, decimals=precision))
        return log_return

    def get_log_return_pdf(self, precision=3):
        log_return = self.__get_log_return(precision)
        log_return_frequency = log_return[0].value_counts()
        log_return_frequency.sort_index(inplace=True)
        log_return_pdf = log_return_frequency / sum(log_return_frequency)
        return log_return_pdf

    def get_log_return_pdf_norm(self, precision=3):
        log_return = self.__get_log_return(precision)
        log_return_frequency = log_return[0].value_counts()
        log_return_frequency.sort_index(inplace=True)
        log_return_pdf = log_return_frequency / sum(log_return_frequency)
        interval_length = 10 ** (-precision)
        return log_return_pdf / sum(log_return_pdf * interval_length)

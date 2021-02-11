#!/usr/bin/env python
# coding: utf-8

"""
File: visualization.py
Class: CIS 422
Date: February 1, 2021
Team: The Nerd Herd
Head Programmer: Jack Sanders
Version 1.0.0

Overview: Visualization and evaluation functions to be used in a Transformation
Tree
"""

# In[287]:


# imports

from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from datetime import datetime
import pandas as pd
from pandas import Series

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest
from scipy.stats import anderson
from sklearn.metrics import mean_squared_error
from math import sqrt

import sklearn.metrics as metrics
from numpy import percentile


# In[288]:

def read_ts(input_file):
    print(type(input_file))
    print(input_file.columns)

def read_matrix(input_file):

    matrix_list = []
    for sublist in input_file:
        for item in sublist:
            matrix_list.append(item)

    df = DataFrame(matrix_list, columns=['Series_old'])
    time = pd.date_range(start=pd.datetime(2000,1,1),periods=len(df))
    time_series = df.assign(Time = time)
    time_series = time_series.assign(Series = time_series['Series_old'])
    del time_series['Series_old']
    time_series.set_index(pd.to_datetime(time_series['Time']))
    return time_series

def csv_to_ts(csv):
    """
    .csv: matrix, y: y-axis/title -> dataframe w/ two columns

    TODO: find function other than assign to pass y in as unique title
    TODO: provide more customizability for periods
    """
    series = read_csv(csv, names=["Series_old"])
    time = pd.date_range(start=pd.datetime(2000,1,1),periods=len(series))
    time_series = series.assign(Time = time)
    time_series = time_series.assign(Series = time_series['Series_old'])
    del time_series['Series_old']
    time_series.set_index(pd.to_datetime(time_series['Time']))
    return time_series


# In[289]:


def plot(series):
    """
    takes one or more TS with Time and Series columns
    For multiple TS dataframes, pass them in as a list of TSs.
    """
    if (type(series) == list):
        for df in series:
            df.plot(x='Time', y='Series')
            pyplot.show()
    else:
        series.plot(x='Time', y='Series')
        pyplot.show()


# In[290]:


def histogram(series):
    """
    takes one or more TS
    For multiple TS dataframes, pass them in as a list of TSs.
    """
    if (type(series) == list):
        for df in series:
            df.hist()
            pyplot.title('Series')
            pyplot.xlabel('categories')
            pyplot.ylabel('values')
            pyplot.show()
    else:
        series.hist()
        pyplot.title('Series')
        pyplot.xlabel('categories')
        pyplot.ylabel('values')
        pyplot.show()


# In[291]:


def summary(series):
    """
    takes one TS
    returns 5 number summary: min, q1, median, q3, max
    """
    # calculate quartiles
    quartiles = percentile(series['Series'], [25, 50, 75])
    # calculate min/max
    data_min, data_max = series['Series'].min(), series['Series'].max()
    # print 5-number summary
    print('Min: %.3f' % data_min)
    print('Q1: %.3f' % quartiles[0])
    print('Median: %.3f' % quartiles[1])
    print('Q3: %.3f' % quartiles[2])
    print('Max: %.3f' % data_max)
    return data_min, quartiles[0], quartiles[1], quartiles[2], data_max

def box_plot(series):
    # box plot
    plt.boxplot(series['Series'])
    plt.show()


# In[292]:


def shapiro_wilk(series):
    """
    takes one TS

    SW tests the null hypothesis that a sample
    came from a normally distributed population.
    """
    # Shapiro-Wilk Test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import shapiro
    # normality test
    stat, p = shapiro(series['Series'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return stat, p


# In[293]:


def d_agostino(series):
    """
    AG tests the null hypothesis that a sample
    came from a normally distributed population.
    """
    # normality test
    stat, p = normaltest(series['Series'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return stat, p


# In[294]:


def anderson_darling(series):
    """
    AD tests the null hypothesis that a sample
    came from a normally distributed population.
    """
    # normality test
    result = anderson(series['Series'])
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    return sl, cv


# In[295]:


def qq_plot(series):
    # QQ Plot
    qqplot(series['Series'], line='s')
    pyplot.show()


# In[296]:


def MSE(actual, forecast):
    """
    Mean Squared Error:
    average squared difference between the
    estimated values and the actual value.
    TODO: edge cases
    """
    mse = mean_squared_error(actual, forecast)
    print('MSE: %f' % mse)
    return mse


# In[297]:


def RMSE(actual, forecast):
    """
    Root Mean Square Error
    standard deviation of the residuals
    TODO: edge cases
    """
    mse = mean_squared_error(actual, forecast)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)


# In[298]:


def MAPE(actual, forecast):
    """
    Mean Absolute Percentage Error (MAPE)
    average absolute percent error for each time period
    minus actual values divided by actual values.
    TODO: edge cases
    """
    actual, forecast = np.array(actual), np.array(forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    print('MAPE: %f' % mape)
    return mape


# In[299]:


def sMAPE(actual, forecast):
    """
    Symmetrical Mean Absolute Percentage Error (sMAPE)
    subtract each actual value from forecast value for each period
    take the root square of the square of the result
    TODO: edge cases
    """
    smape = 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast))*100)
    print('sMAPE: %f' % smape)


# In[300]:


# In[ ]:

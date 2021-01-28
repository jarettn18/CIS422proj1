"""
File: preprocessing.py
Class: CIS 422
Date: January 21, 2021
Team: The Nerd Herd
Head Programmer: Logan Levitre
Version 0.1.7

Overview: Preprocessing functions to be used with Time Series Data.
    -- Need second opinion for revision of header structure --
"""
import pandas as pd
import math
from patsy.highlevel import dmatrix
import numpy as np
import datetime as dt
# for Clips function - user must pip install pyjanitor
#import janitor as pyj


def read_from_file(input_file):
    """
    Function reads TimeSeries csv file and stores data
    as a DataFrame object
    :param input_file: file passed to function containing Time Series Data
    :return: returns DataFrame object declared time_series
    """
    time_series = pd.read_csv(input_file)
    return time_series


def write_to_file(output_file, ts):
    """
    Takes Series and outputs the data into the output_file
    :param output_file: file to store data in
    :param ts: time_series to be stored
    :return: None
    """
    ts.to_csv(output_file, index=False)


def denoise(time_series):
    """
    Removes noise from a time series. Produces a time series with less noise than
    the original one.
    :param time_series: Time series data
    :return: returns a new Time Series with less noise
    """
    clean_time_series = impute_missing_data(time_series)
    clean_time_series = impute_outliers(clean_time_series)
    return clean_time_series


def impute_missing_data(time_series):
    """
    Corrects missing data entries inside time_series
    takes value to the right and uses that as placeholder
    for the data missing - if multiple entries are missing - skip
    :param time_series: Time series data
    :return: time series with filled in missing values
    """
    # Possibly use .shape - Return a tuple representing the dimensionality of the DataFrame.
    restored_series = time_series.copy()
    # find NaN and fill with data to the right of it
    restored_series = restored_series.fillna(method='ffill')
    return restored_series


def impute_outliers(time_series):
    """
    Find and remove outliers within the Time series
    - similar to impute_missing_data functions -
    :param time_series: Time series data
    :return: concise Time series without outliers
    """
    # create new time series w/o outliers
    ts_without = time_series.copy()
    # get last column location
    data_col = time_series.columns[len(ts_without.columns) - 1]
    #row = time_series.index
    # get high quartile
    quantile_high = ts_without[data_col].quantile(0.95)
    # get low end quartile
    quantile_low = ts_without[data_col].quantile(0.05)
    # go through data in time_series
    for ind in ts_without.index:
        # if value is outside quartile's delete it
        if ts_without.at[ind, data_col] < quantile_low or ts_without.at[ind, data_col] > quantile_high:
            # drop specific rows date, timestamp & value
            # axis=1 specifies Rows
            ts_without.drop([ind], axis=0, inplace=True)
    return ts_without


def longest_continuous_run(time_series):
    """
    Isolates the most extended portion of the time series without
    missing data.
    :param time_series: Time series data
    :return: new a time series without any missing data or outliers
    """
    # copy time_series
    longest_run_ts = time_series.copy()
    # DISCLAIMER: Code lines 92-101 Referenced from
    # https://stackoverflow.com/questions/41494444/pandas-find-longest-stretch-without-nan-values
    # get data values and store as array
    data_col = longest_run_ts.columns[len(longest_run_ts.columns) - 1]
    lr_index = longest_run_ts.index[pd.isna(longest_run_ts[data_col])].tolist()
    # find difference between index of TS
    # if lr_index is empty then there are no NaN's
    if len(lr_index) > 0:
        diff = lr_index[1] - lr_index[0]
        # placeholder for start,stop indexes
        first_idx = 0
        last_idx = 1
        # loop through array getting difference of consecutive values
        for idx in range(1, len(lr_index) - 1):
            if lr_index[idx + 1] - lr_index[idx] > diff:
                diff = lr_index[idx + 1] - lr_index[idx]
                first_idx = lr_index[idx]
                last_idx = lr_index[idx + 1]
        # get start/stop times from TS at indexed locations
        time_col = time_series.columns[0]
        start_time = time_series.at[first_idx + 1, time_col]
        end_time = time_series.at[last_idx - 1, time_col]
        # clip time_series from start to stopping point to get longest run
        longest_run_ts = clip(time_series, start_time, end_time)
        return longest_run_ts
    return longest_run_ts


def clip(time_series, starting_date, final_date) -> object:
    """
    clips the time series to the specified period’s data.
    :param time_series: Time series data
    :param starting_date: first day to be included in new TS
    :param final_date: last date to be included in  new TS
    :return: a portion of the time series from start_date to final_date
    """
    # copy ts into new obj
    # get csv name for column with dates
    dates = time_series.columns[0]
    clipped = time_series.copy()
    # call filter_date function to get dates/values
    #filtered = clipped.filter_date(dates, starting_date, final_date)
    # return time frame
    #return filtered


def assign_time(time_series, start, increment):
    """
    If we do not have the times associated with a sequence of readings.
    Start at a specific time and increment for the following entry.
    :param time_series: Time series data
    :param start: beginning of the timestamp
    :param increment: difference to add to next timestamp
    :return: a new time_series with timestamps assigned to each entry
    """
    new_series = time_series.copy()
    new_series.insert(loc=0, column='Timestamp: Hour', value='')
    column_list = new_series.columns[0]
    # MM/DD/YYYY
    date_splice = start.split(sep="/")
    date_year = int(date_splice[2])
    date_month = int(date_splice[0])
    date_day = int(date_splice[1])
    date = dt.datetime(date_year, date_month, date_day)
    for idx in new_series.index:
        new_series.at[idx, column_list] = date
        date += dt.timedelta(hours=increment)

    return new_series


def difference(time_series):
    """
    Takes the time series data and transforms each entry
    as the difference between it and the next entry
    :param time_series: Time series data
    :return: time series containing the difference between each original entry
    """
    ts_difference = time_series.copy()
    # get column index
    data_col = time_series.columns[len(ts_difference.columns) - 1]
    # loop through column
    for ind in ts_difference.index:
        # if index is the last entry
        if ind == (len(ts_difference.index) - 1):
            # assign index the value of 0 as there is no difference between next entry
            val = time_series.at[ind, data_col] - time_series.at[ind, data_col]
            ts_difference.at[ind, data_col] = val
        else:
            # else assign entry the value of consecutive - current
            val = time_series.at[ind + 1, data_col] - time_series.at[ind, data_col]
            ts_difference.at[ind, data_col] = val
    return ts_difference


def scaling(time_series):
    """
    Produces a time series whose magnitudes are scaled so that the resulting
    magnitudes range in the interval [0,1].
    :param time_series: Time series data
    :return: a new time series with magnitudes between 0 - 1
    """
    data_col = time_series.columns[len(time_series.columns) - 1]
    normalized_ts = time_series.copy()
    normalized_column = (time_series - time_series.mean()) / time_series.std()
    normalized_ts[data_col] = normalized_column[data_col].values
    return normalized_ts


def standardize(time_series):
    """
    Produces a time series whose mean is 0 and variance is 1.
    :param time_series: Time series data
    :return: a Time series with mean of 0/variance is 1
    """
    standard_ts = time_series.copy()
    data_col = time_series.columns[len(time_series.columns) - 1]
    a, b = 10, 50
    x, y = standard_ts[data_col].min(), standard_ts[data_col].max()
    standard_ts[data_col] = (standard_ts[data_col] - x) / (y - x) * (b - a)
    return standard_ts


def logarithm(time_series):
    """
    scales the time series, all entries become log_10 values of initial
    values within the Time Series.
    :param time_series: Time series data
    :return: returns scaled time series containing log_10 results of generates
     a version of previous values
    """
    log_10_time_series = time_series.copy()
    # get column name containing data
    data_col = time_series.columns[len(log_10_time_series.columns) - 1]
    # loop through rows of TS
    for idx in log_10_time_series.index:
        # go through each row of last column - assign cubed root of value at each
        # index of the column
        log_10_time_series.at[idx, data_col] = math.log10(log_10_time_series.at[idx, data_col])
    return log_10_time_series


def cubic_roots(time_series):
    """
    scales the time series where all entries are
    the cubic root of the initial values. - Assume last column contains Data
    :param time_series: Time series data
    :return: a new time series with entries being
            the cubic root of previous values
    """
    cubed_time_series = time_series.copy()
    # get column name containing data
    data_col = time_series.columns[len(cubed_time_series.columns) - 1]
    # loop through rows of TS
    for idx in cubed_time_series.index:
        # go through each row of last column - assign cubed root of value at each
        # index of the column - idx = row
        cubed_time_series.at[idx, data_col] = math.pow(cubed_time_series.at[idx, data_col], 1 / 3)
    return cubed_time_series


def split_data(time_series, perc_training, perc_valid, perc_test):
    """
    Splits a time series into
    training, validation, and testing according to the given percentages.
    :param time_series: Time series data
    :param perc_training: percentage of time series data to be used for training
    :param perc_valid: percentage of time series data to be used for validation
    :param perc_test: percentage of time series data to be used for testing
    :return: None
    """
    ts_size = len(time_series.index)
    test_perc = (perc_test / 100)
    test_valid = (perc_valid / 100)
    test_training = (perc_training / 100)
    t_p_l = int((ts_size * test_perc))
    t_v_l = int((ts_size * test_valid))
    t_t_l = int((ts_size * test_training))
    p_tr_cut = time_series.iloc[:t_t_l, :]
    p_tr_cut.to_csv("perc_training.csv", index=False)
    p_v_cut = time_series.iloc[t_t_l + 1:(t_v_l + t_t_l), :]
    p_v_cut.to_csv("perc_valid.csv", index=False)
    p_test_cut = time_series.iloc[t_p_l + 1:ts_size, :]
    p_test_cut.to_csv("perc_test.csv", index=False)


def design_matrix(time_series, input_index=0, output_index=0):
    """
    Creates a matrix of time series data
    takes an input idx - and returns output index
    :param time_series: Time series data
    :param input_index: index of chosen data
    :param output_index: Y val after being put into matrix
    :return: a numpy Matrix of touples
    """
    # are we to create/ find algo that takes input and makes it output?
    tmp_ts = time_series.copy()
    time_col = tmp_ts.columns[0]
    mst_col = tmp_ts.columns[len(tmp_ts.columns) - 2]
    # remove time column - not necessary
    # axis=1 specifies Columns
    tmp_ts.drop([time_col], axis=1, inplace=True)
    tmp_ts.drop([mst_col], axis=1, inplace=True)
    # create patsy dmatrix using formula for linear regression
    # passing time series data into it - returns a matrix
    # Convert TS to numpy array - Matrix
    ts_matrix = tmp_ts.to_numpy()
    # next line possible for function application to matrix
    # a_matrix = dmatrix("y ~ a + bX", data=tmp_ts, return_type="dataframe")
    return ts_matrix


def design__matrix(time_series, m_i, t_i, m_0, t_0):
    """
    Creates a Matrix up to certain position of Time series
    depends on m_i & t_i
    :param time_series: Time series data
    :param m_i: magnitude at index I - input
    :param t_i: timestamp at index I - change
    :param m_0: magnitude of index 0 - predicted mag
    :param t_0: timestamp of index 0 - predicted time
    :return: Matrix of time series data up to m_i & t_i
    """
    # Needs further Discussion with Zeke/Jarett
    pass


def ts2db(input_file, perc_training, perc_valid, perc_test, input_index,
          output_index, output_file):
    """
    combines reading a file, splitting the
    data, converting to database, and producing the training databases.
    :param input_file: file to be read and split
    :param perc_training: percentage of data to be split into training data
    :param perc_valid: percentage of data to be split into valid data
    :param perc_test: percentage of the data to be split into test data
    :param input_index: initial index of data
    :param output_index: index to take data from - looks similar to input index
    :param output_file: file for data to be written into
    :return: Writes Matrix to csv file
    """
    ts = read_from_file(input_file)
    split_data(time_series=ts, perc_training=perc_training, perc_test=perc_test, perc_valid=perc_valid)
    ts_training = read_from_file("perc_training.csv")
    matrix = design_matrix(ts_training)
    dt_matrix = pd.DataFrame(matrix, columns=['Data'])
    dt_matrix.drop(dt_matrix.index[0], axis=0, inplace=True)
    write_to_file(output_file, dt_matrix)



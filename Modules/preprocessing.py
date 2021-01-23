"""
File: preprocessing.py
Class: CIS 422
Date: January 21, 2021
Team: The Nerd Herd
Head Programmer: Logan Levitre
Version 0.1.1

Overview: Preprocessing functions to be used with Time Series Data.
    -- Need second opinion for revision of header structure --
"""
import pandas as pd
import math
import numpy as np


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
    clean_time_series = difference(time_series)
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
    column = restored_series.columns - 1
    for idx in range(len(restored_series)):
        if restored_series.loc[idx, column].isna():
            if restored_series.loc[idx + 1, column].isna():
                # if sequential Na's exist take value of 4 rows ahead
                restored_series.loc[idx, column].fillna(restored_series.loc[idx + 4, column])
            # if singular Na, use data of next entry as a fill
            else:
                restored_series.loc[idx, column].fillna(restored_series.loc[idx + 1, column])
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
    # get high quartile
    quantile_high = ts_without[data_col].quantile(0.98)
    # get low end quartile
    quantile_low = ts_without[data_col].quantile(0.02)
    # go through data in time_series
    for idx in range(len(ts_without)):
        # if value is outside quartile's delete it
        if ts_without[idx, data_col] < quantile_low or ts_without[idx, data_col] > quantile_high:
            # drop specific rows date, timestamp & value
            ts_without.drop([idx])
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
    ts_values = longest_run_ts.data_col.values
    # create mask of array as boolean
    mask = np.concatenate(([True], np.isnan(ts_values), [True]))
    # take out negative values if any
    start_stop_limits = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1, 2)
    # get start/stop indexes of longest run - subtract 1 from stop to get correct
    # index of stop location
    start, stop = start_stop_limits[(start_stop_limits[:, 1] - start_stop_limits[:, 0]).argmax()]
    # clip time_series from start to stopping point to get longest run
    longest_run_ts = clip(time_series, start, stop - 1)
    return longest_run_ts


def clip(time_series, starting_date, final_date):
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
    # filter time series looking at dates
    mask = (time_series[dates] >= starting_date) \
           & (time_series[dates] <= final_date)
    # assign clipped_time_series the rows of dates within time frame
    clipped_time_series = time_series.loc[mask]
    # return time frame
    return clipped_time_series


def assign_time(time_series, start, increment):
    """
    If we do not have the times associated with a sequence of readings.
    Start at a specific time and increment for the following entry.
    :param time_series: Time series data
    :param start: beginning of the timestamp
    :param increment: difference to add to next timestamp
    :return: a new time_series with timestamps assigned to each entry
    """
    pass


def difference(time_series):
    """
    Takes the time series data and transforms each entry
    as the difference between it and the next entry
    :param time_series: Time series data
    :return: time series containing the difference between each original entry
    """
    ts_difference = time_series.copy()
    data_col = time_series.columns[len(ts_difference.columns) - 1]
    for ind in ts_difference.index:
        ts_difference.at[ind, data_col] = 0
    #for index, row in time_series.iterrows():
    #    ts_difference.loc[data_col, index] = 6
    #ts = ts_difference[data_col].diff()
    return ts_difference
    #for index, idx in ts_difference.iterrows():
    #    if index is not len(ts_difference.index) - 1:
    #        ts_difference.replace() = time_series.iloc[idx + 1, data_col] - time_series.iloc[idx, data_col]
    #return ts_difference


def scaling(time_series):
    """
    Produces a time series whose magnitudes are scaled so that the resulting
    magnitudes range in the interval [0,1].
    :param time_series: Time series data
    :return:
    """
    pass


def standardize(time_series):
    """
    Produces a time series whose mean is 0 and variance is 1.
    :param time_series: Time series data
    :return: a Time series with mean of 0/variance is 1
    """
    pass


def logarithm(time_series):
    """
    scales the time series, all entries become log_10 values of initial
    values within the Time Series.
    :param time_series: Time series data
    :return: returns scaled time series containing log_10 results of generates
     a version of previous values
    """
    log_10_time_series = time_series.copy()
    data_col = time_series.columns[len(log_10_time_series.columns) - 1]
    for idx in range(len(log_10_time_series)):
        # go through each row of last column - assign cubed root of value at each
        # index of the column
        log_10_time_series.loc[idx, data_col] = math.log10(log_10_time_series.loc[idx, data_col])
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
    data_col = time_series.columns[len(cubed_time_series.columns) - 1]
    for idx in range(len(cubed_time_series)):
        # go through each row of last column - assign cubed root of value at each
        # index of the column - idx = row
        cubed_time_series.loc[idx, data_col] = math.pow(cubed_time_series.loc[idx, data_col], 1 / 3)
    return cubed_time_series


def split_data(time_series, perc_training, perc_valid, perc_test):
    """
    Splits a time series into
    training, validation, and testing according to the given percentages.
    :param time_series: Time series data
    :param perc_training: percentage of time series data to be used for training
    :param perc_valid: percentage of time series data to be used for validation
    :param perc_test: percentage of time series data to be used for testing
    :return: multiple csv files holding training, valid, test values
    """
    pass


def design_matrix(time_series, input_index, output_index):
    """
    Creates a matrix of time series data
    :param time_series: Time series data
    :param input_index: Unknown as of 1/20
    :param output_index: Unknown as of 1/20
    :return: Matrix of time series data
    """
    pass


def design__matrix(time_series, m_i, t_i, m_0, t_0):
    """
    Creates a Matrix up to certain position of Time series
    depends on m_i & t_i
    :param time_series: Time series data
    :param m_i: magnitude at index I
    :param t_i: timestamp at index I
    :param m_0: magnitude of index 0
    :param t_0: timestamp of index 0
    :return: Matrix of time series data up to m_i & t_i
    """
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
    :param input_index: initial index for matrix
    :param output_index: index to take data from
    :param output_file: file for data to be written into
    :return: multiple csv files and a matrix
    """
    pass

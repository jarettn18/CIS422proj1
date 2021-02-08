"""
File: preprocessing.py
Class: CIS 422
Date: February 2, 2021
Team: The Nerd Herd
Head Programmer: Logan Levitre
Version 1.0.5

Overview: Preprocessing functions to be used with Time Series Data.
"""
import pandas as pd
from math import pow, log10
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime as dt


# import janitor as pyj


def read_from_file(input_file):
    """
    Function reads TimeSeries csv file and stores data
    as a DataFrame object
    :param input_file: file passed to function containing Time Series Data
    :return: returns DataFrame object declared time_series
    """
    time_series = pd.read_csv(input_file, parse_dates=True)
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
    data_col = time_series.columns[len(restored_series.columns) - 1]
    restored_series[data_col].replace(0, restored_series[data_col].mean().round(5), inplace=True)
    #restored_series[data_col].replace(0, np.nan, inplace=True)
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
    # row = time_series.index
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
    # DISCLAIMER: Code lines 92-101 Referenced from
    # https://stackoverflow.com/questions/41494444/pandas-find-longest-stretch-without-nan-values
    # get data values and store as array
    longest = time_series.copy()
    data_col = longest.columns[len(longest.columns) - 1]
    longest[data_col].replace(0, np.nan, inplace=True)
    print(longest)
    lr_index = longest.index[pd.isna(longest[data_col])].tolist()
    # find difference between index of TS
    # if lr_index is empty then there are no NaN's
    if len(lr_index) > 0:
        diff = lr_index[1] - lr_index[0]
        #    # placeholder for start,stop indexes
        first_idx = 0
        last_idx = 1
        #    # loop through array getting difference of consecutive values
        for idx in range(0, len(lr_index)):
            if idx == (len(lr_index) - 1):
                if len(longest.index) - lr_index[idx] > diff:
                    diff = len(longest.index) - lr_index[idx]
                    first_idx = lr_index[idx] + 1
                    last_idx = (len(longest) - 1)
            elif lr_index[idx + 1] - lr_index[idx] > diff:
                diff = lr_index[idx + 1] - lr_index[idx]
                first_idx = lr_index[idx]
                last_idx = lr_index[idx + 1]
        # get start/stop times from TS at indexed locations
        # create new DataFrame of sliced portion
        clipped_data = longest.loc[first_idx:last_idx]
        return clipped_data
    return time_series


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
    filtered = clipped.filter_date(dates, starting_date, final_date)
    # return time frame
    return filtered


def assign_time(time_series, start, increment):
    """
    If we do not have the times associated with a sequence of readings.
    Start at a specific time and increment for the following entry.
    :param time_series: Time series data
    :param start: beginning of the timestamp
    :param increment: difference to add to next timestamp
    :return: a new time_series with timestamps assigned to each entry
    """
    data_col = time_series.columns[len(time_series.columns) - 1]
    new_ts = pd.DataFrame(columns=['Timestamp: Hour:', 'Data'])
    for x in time_series.index:
        new_ts.loc[x] = time_series.at[x, data_col]

    column_list = new_ts.columns[0]
    # MM/DD/YYYY - get each segment of input to create datetime obj
    date_splice = start.split(sep="/")
    date_year = int(date_splice[2])
    date_month = int(date_splice[0])
    date_day = int(date_splice[1])
    date = dt.datetime(date_year, date_month, date_day)
    # insert time into DataFrame
    for idx in new_ts.index:
        new_ts.at[idx, column_list] = date
        date += dt.timedelta(hours=increment)

    return new_ts


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
    normalized_ts = time_series.copy()
    data_col = time_series.columns[len(time_series.columns) - 1]
    data = [x for x in normalized_ts[data_col]]
    new_data = pd.DataFrame(data, columns=[data_col])

    scalar = MinMaxScaler()
    normalized_ts_new = (scalar.fit_transform(new_data))

    for idx in range(len(normalized_ts)):
        if idx == len(normalized_ts):
            break
        normalized_ts[data_col].values[idx] = normalized_ts_new[idx]
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
        log_10_time_series.at[idx, data_col] = log10(log_10_time_series.at[idx, data_col])
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
        cubed_time_series.at[idx, data_col] = pow(cubed_time_series.at[idx, data_col], 1 / 3)
    return cubed_time_series


def split_data(time_series, perc_training=50, perc_test=50):
    """
    Splits a time series into
    training and testing according to the given percentages.
    :param time_series: Time series data
    :param perc_training: percentage of time series data to be used for training
    :param perc_test: percentage of time series data to be used for testing
    :return train_ts: Return the Training Time Series
	:return test_ts: Return the Testing Time series
    """
    # Create error handling for if percentages are of same val for equal comparison
    if perc_training + perc_test == 100:
        # Get length of ts
        ts_size = len(time_series)
        # get percentage val to be perc
        test_perc = (perc_test / 100)
        # get percentage val to be training
        test_training = (perc_training / 100)
        # size of training Data
        t_p_l = int((ts_size * test_perc))
        # size of training Data
        t_t_l = int((ts_size * test_training))
        print(t_p_l)
        print(t_t_l)
        # cut TS based on percentage for Training
        train_ts = time_series[0:t_p_l]
        # cut TS based on percentage for Testing
        test_ts = time_series[t_p_l:]

        return train_ts, test_ts
    else:
        print("Error: Training percentage and Test Percentage "
              + "must sum to 100")


def design_matrix(time_series, prev_index):
    """
    Creates a matrix of time series data
    while adding to input/output index
    :param time_series: Time series data
    #:param input_index: indices of training data for error
    :param prev_index: forecasted index for error testing
    :return: a numpy Matrix data
    """
    # BEFORE TAKING TIME AWAY
    # are we to create/ find algo that takes input and makes it output?
    inputs = []
    tmp_ts = time_series.copy()
    columns = len(tmp_ts.columns)

    # data_col = tmp_ts.columns[len(tmp_ts.columns) - 1]

    index = 0
    # t = len(tmp_ts)'
    for i in range(len(tmp_ts.values)):
        inputs.append([i + prev_index])
        index = i
    # remove time column - not necessary
    # axis=1 specifies Columns
    if columns == 2:
        tmp_ts.drop([tmp_ts.columns[0]], axis=1, inplace=True)
    if columns == 3:
        # remove time column
        tmp_ts.drop([tmp_ts.columns[1]], axis=1, inplace=True)
        # remove MST/Other column
        tmp_ts.drop([tmp_ts.columns[0]], axis=1, inplace=True)
    # create patsy dmatrix using formula for linear regression
    # passing time series data into it - returns a matrix
    # Convert TS to numpy array - Matrix
    # convert data to array
    ts_matrix = tmp_ts.to_numpy()
    ts_matrix = ts_matrix.reshape(1, -1)
    ts_matrix = ts_matrix[0]
    return ts_matrix, inputs, index


def design__matrix(time_series, m_i, t_i, m_O, t_O):
    """
    Creates a Matrix up to certain position of Time series
    depends on m_i & t_i
    :param time_series: Time series data
    :param m_i: number of readings used from Training
    :param t_i: distance between each index in used from training
    :param m_O: number of output readings for prediction
    :param t_O: ticks between output readings in prediction
    :return: Matrix of time series data
    """
    ts_train = time_series.copy()
    # convert m_i, t_i into input index
    # convert m_O, t_O into output index
    inputs = []
    outputs = []
    # number of input indexes to take
    length_in = m_i
    # spaces apart
    space_in = t_i
    t = (len(ts_train) // 2)
    for idx in reversed(range(0, t, space_in)):
        if len(inputs) == length_in:
            break
        else:
            inputs.append(idx)
    length_o = m_O
    # spaces apart
    space_o = t_O
    for x in range(t, len(ts_train), space_o):
        if len(outputs) == length_o:
            break
        else:
            outputs.append(x)
    # inputs now has array of indexes spaced apart t_i
    # outputs now has array of indexes spaced apart t_O
    matrix = design_matrix(ts_train, inputs, outputs)
    # num of output indexes to take
    # take the values of input/output index and create matrix to return
    return matrix


def ts2db(input_file_train, input_file_test=None):
    """
    :param input_file_train: Training file to be read (and split if necessary)
	:param input_file_test: Testing file to be read
    :return train_ts: Training Time Series for Validation
    :return train_inputs: Input Time Series for Training
    :return test_ts: Test Time Series for Forecast Accuracy
    :return test_inputs: Input Time Series for Forecast
    """
	#Denoise The Training Data
    data = read_from_file(input_file_train)
    denoised_data = denoise(data)

	#If Test File passed in, Training Data does not need to be split
    if (input_file_test is None):
        train_data, test_data = split_data(denoised_data, 50, 50)
    else:
        test = read_from_file(input_file_test)
        test_data = denoise(test)
        train_data = denoised_data
	
	#Create Inputs and Time Series
    train_ts, train_inputs, prev_i = design_matrix(train_data, 0)
    test_ts, test_inputs, ignore_this = design_matrix(test_data, prev_i + 1)
	
    return train_ts, train_inputs, test_ts, test_inputs

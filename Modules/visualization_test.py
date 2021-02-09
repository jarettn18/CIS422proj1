"""
File: visualization_test.py
Class: CIS 422
Date: February 9, 2021
Team: The Nerd Herd
Head Programmer: Jack Sanders
Version 1.0.0

Overview: Testing for visualization and evaluation module
"""

import visualization as viz

def testing(csv):
    # establish series variable to be used throughout
    series = viz.csv_to_ts(csv, 'Temperature')
    print("\n----data frame head----")
    print(series.head())
    print("\n----data frame column types----")
    print(series.dtypes)
    print("\n----plot----")
    viz.plot(series)
    print("\n----histogram----")
    viz.histogram(series)
    print("\n----5 number summary----")
    viz.summary(series)
    print("\n----box plot----")
    viz.box_plot(series)
    print("\n----Shapiro-Wilk Normality test----")
    sw_stats = viz.shapiro_wilk(series)
    print("\n----d'agostino Normality test----")
    ag_stats = viz.d_agostino(series)
    print("\n----Quantile-Quantile Plot----")
    viz.qq_plot(series)
    print("\n----Anderson Darling Normality test----")
    ad_stats = viz.anderson_darling(series)
    print("\n----Mean Squared Error----")
    actual = [0.0, 0.5, 0.0, 0.5, 0.0]
    forecast = [0.2, 0.4, 0.1, 0.6, 0.2]
    mse_stats = viz.MSE(actual, forecast)
    print("\n----Root Mean Square Error----")
    actual = [0.0, 0.5, 0.0, 0.5, 0.0]
    forecast = [0.2, 0.4, 0.1, 0.6, 0.2]
    rmse_stats = viz.RMSE(actual, forecast)
    print("\n----Mean Absolute Percentage Error----")
    actual = [12, 13, 14, 15, 15,22, 27]
    forecast = [11, 13, 14, 14, 15, 16, 18]
    mape_stats = viz.MAPE(actual, forecast)
    print("\n----Symmetrical Mean Absolute Percentage Error----")
    actual = np.array([12, 13, 14, 15, 15,22, 27])
    forecast = np.array([11, 13, 14, 14, 15, 16, 18])
    smape_stats = viz.sMAPE(actual, forecast)

def main():
    # test on one-dimensional input
    csv = '../TestData/1_temperature_train.csv'
    testing(csv)

if __name__ == '__main__':
    main()

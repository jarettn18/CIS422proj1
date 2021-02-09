"""
File: main.py
Class: CIS 422
Date: January 21, 2021
Team: The Nerd Herd
Head Programmers: Logan Levitre, Jarett Nishijo
Version 0.1.0

Overview: Main file for Machine Learning Models + Statistics Visualization
"""

import preprocessing as prep
import model as mp
import pandas as pd
import numpy as np
import math
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def main():

	#Get File Path
	current_path = os.path.dirname(os.getcwd())
	dir_list = os.listdir(current_path + "/TestData/")
	dir_list.sort()

	for i in range(0, 10, 2):
		fname_test = current_path + "/TestData/" + dir_list[i]
		fname = current_path + "/TestData/" + dir_list[i+1]

		#Read and Preprocess Data from File
		ts, inputs, ts_test, inputs_test = prep.ts2db(fname, fname_test)

		#Create and Train Model
		rf = mp.rf_model()
		mlp = mp.mlp_model()

		rf.fit(inputs, ts)
		mlp.fit(inputs, ts)

		forecast_rf = rf.forecast(inputs_test)
		forecast_mlp = mlp.forecast(inputs_test)

		print("################# RF FORECAST FOR", dir_list[i+1], "##############################")
		print(forecast_rf[0:100])
		print("################# MLP FORECAST FOR", dir_list[i+1], "##############################")
		print(forecast_mlp[0:100])
		print("################# TEST RESULTS FOR", dir_list[i+1], "##############################")
		print(ts_test[0:100])
		print("\n")

if __name__ == '__main__':
	main()

"""
File: main.py
Class: CIS 422
Date: January 21, 2021
Team: The Nerd Herd
Head Programmers: Logan Levitre, Zeke Peterson, Jarett Nishijo, Callista West, Jack Sanders
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
	
	#for i in range(0, 10, 2):
	fname_test = current_path + "/TestData/" + dir_list[0]
	fname = current_path + "/TestData/" + dir_list[0+1]

	#Read and Preprocess Data from File
	"""
	test = prep.read_from_file(fname_test)
	denoised_test = prep.denoise(test)

	data = prep.read_from_file(fname)
	denoised_data = prep.denoise(data)

	ts, inputs, prev_i = prep.design_matrix(denoised_data, 0)
	ts_test, inputs_test, ignore = prep.design_matrix(denoised_test, prev_i + 1)
	"""
	
	ts, inputs, ts_test, inputs_test = prep.ts2db(fname, fname_test)

	#Create and Train Model
	rf = mp.rf_model()
	mlp = mp.mlp_model()
	
	rf.fit(inputs, ts)
	mlp.fit(inputs, ts)

	forecast_rf = rf.forecast(inputs_test)
	forecast_mlp = mlp.forecast(inputs_test)

	print(inputs[0:100])
	print(ts[0:100])

	print(forecast_mlp[0:100])

	"""
	print("################# RF FORECAST FOR", dir_list[i+1], "##############################")
	print(forecast_rf[0:100])
	print("################# MLP FORECAST FOR", dir_list[i+1], "##############################")
	print(forecast_mlp[0:100])
	print("################# TEST RESULTS FOR", dir_list[i+1], "##############################")
	print(ts_test[0:100])
	print("\n")
	"""
if __name__ == '__main__':
	main()
"""
MLP MODEL TESTING
"""
"""
# PRODUCE DATA SETS
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

#TRAIN MODEL
mlp = mp.mlp_model()
mlp.fit(X_train, y_train)

#FORECAST DATA
forecast = mlp.forecast(X_test)
print("MLP Targets")
print(y_test)
print("MLP Forecast")
print(str(forecast) + "\n")

#ACCURACY SCORE
score = mlp.model.score(X_test, y_test)
print("MLP Accuracy: "+ str(score) + "\n")

#RF MODEL TESTING
#TRAIN MODEL
rf = mp.mlp_model()
rf.fit(X_train, y_train)

#FORECAST DATA
forecast = mlp.forecast(X_test)
print("RF Targets")
print(y_test)
print("RF Forecast")
print(str(forecast) + "\n")

#ACCURACY SCORE
score = mlp.model.score(X_test, y_test)
print("RF Accuracy: "+ str(score))
"""

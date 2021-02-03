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

"""
@EVERYONE DELETE/IMPLEMENT CODE AS NEEDED
"""

def main():

	#Get File Path
	current_path = os.path.dirname(os.getcwd())
	fname = current_path + "/TestData/1_temperature_train.csv"
	fname_test = current_path + "/TestData/1_temperature_test.csv"

	#Read and Preprocess Data from File
	test = prep.read_from_file(fname_test)
	denoised_test = prep.denoise(test)

	data = prep.read_from_file(fname)
	denoised_data = prep.denoise(data)

	ts, inputs, prev_i = prep.design_matrix(denoised_data, 0)
	ts_test, inputs_test, ignore = prep.design_matrix(denoised_test, prev_i + 1)


	#Create and Train Model

	mlp = mp.rf_model()
	mlp.fit(inputs, ts)
	forecast = mlp.forecast(inputs_test)
	print(ts[0:100])
	print(forecast[0:100])
	print(ts_test[0:100])



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

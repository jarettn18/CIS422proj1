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
import math
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

"""
@EVERYONE DELETE/IMPLEMENT CODE AS NEEDED
"""

def main():

	#Get File Path
	current_path = os.path.dirname(os.getcwd())
	fname = current_path + "/TestData/WindSpeed2010Jan20mMin.csv"

	#Read and Preprocess Data from File
	data = prep.read_from_file(fname)
	denoised_data = prep.denoise(data)

	#Write to intermediary training file
	prep.write_to_file("data_denoised.csv", denoised_data)

	#Transform Time Series data to Data base
	inputs, outputs = [], []

	db = prep.ts2db("data_denoised.csv", 50, 25, 25, inputs, outputs, "outputs.csv")
	train_data_unprepped = prep.read_from_file("perc_training.csv")
	valid_data_unprepped = prep.read_from_file("perc_valid.csv")
	test_data_unprepped = prep.read_from_file("perc_test.csv")

	train_data = prep.design_matrix(train_data_unprepped, inputs, outputs)
	valid_data = prep.design_matrix(valid_data_unprepped, inputs, outputs)
	test_data = prep.design_matrix(test_data_unprepped, inputs, outputs)


	print(len(train_data))
	print(len(test_data))
	print(len(valid_data))
		

	mlp = mp.mlp_model()
	mlp.fit(train_data, valid_data)
	forecast = mlp.forecast(test_data)
	print(forecast)
	
	"""
	data_op = prep.read_from_file(fname)
	denoised_data = prep.denoise(data_op)

	print(denoised_data)
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

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
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

"""
@EVERYONE DELETE/IMPLEMENT CODE AS NEEDED
"""

"""
MLP MODEL TESTING
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

"""
RF MODEL TESTING
"""
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

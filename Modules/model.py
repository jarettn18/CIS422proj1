"""
File: model.py
Class: CIS 422
Date: January 20, 2021
Team: The Nerd Herd
Head Programmer: Jarett Nishijo
Version 0.1.0

Overview: Modeling functions to be used with preprocessed
        data from preprocessing.py
"""
import preprocessing
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class mlp_model():

    def __init__(self, input_dimension=2, output_dimension=1, layers=100):
        """
        Initializes an MLP model with SKLearn's default params
        :param input_dimension: ?
        :param output_dimension: ?
        :param layers: The ith element represents the number of neronds in the ith hidden layer
        """
        self.model = MLPClassifier(hidden_layer_sizes=layers)
        self.input_d = input_dimension
        self.output_d = output_dimension

    def classify(self, samples=100, rand_state=1):
        """
        Tester function to produce classified data
        NOTE: train_test_split(x,y) needs to be run on data before fitting into MLP model
        :param samples: n_samples to be produced
        :param rand_state: random state of ith variable
        :return x_train: trained data set over x matrix
        :return x_test: test data from x matrix (to be used with forecast)
        :return y_train: trained data set over y matrix
        :return y_test: test data from y matrix
        """
        X, y = make_classification(n_samples=samples, random_state=rand_state)
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=rand_state)

        return x_train, x_test, y_train, y_test

    def fit(self, x_train, y_train):
        """
        Fits the model to data matrix x and target y
        :param x_train: input data
        :param y_train: target values
        :return: None
        """
        self.model.fit(x_train, y_train)

    def forecast(self, x):
        """
        Produces a forecast for the values of sparse matrix X
        :param x: Input data
        :return: The predicted classes
        """
        forecast = self.model.predict(x)

        return forecast

class rf_model():

    def __init__(self):
        """
        Initalizes a Random Forest Model with SKLearn's default params
        """
        self.model = RandomForestClassifier()

    def fit(self, x_train, y_train):
        """
        Fits the model to data matrix x and target y
        :param x_train: input data
        :param y_train: target values
        :return: None
        """
        self.model.fit(x_train, y_train)

    def forecast(self, x):
        """
        Produces a forecast for the values of sparse matrix X
        :param x: Input data
        :return: The predicted classes
        """
        forecast = self.model.predict(x)

        return forecast

import sys
import numpy as np
import matplotlib.pyplot as mpl
import csv
import pandas as pd
from numpy import array
import math
import random


class LogisticRegression:

	def __init__(self, rate, iterations):
		self.rate = rate
		self.iterations = iterations		

	def sigmoid(self, x):
		z = np.exp(-x)
		return 1 / (1 + z)


	def loss(self, sigmoid, y):
		return (-y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid)).mean()

	def fit (self, X, Y, weights):
		news = weights
		for _ in range(self.iterations):
			total_loss = np.zeros_like(weights)
			weights = news

			for i in range(len(X)):
				difference = Y[i] - self.sigmoid(np.dot(weights, X[i]))
				total_loss = np.add(total_loss, X[i] * difference)
			
			news = np.add(weights, self.rate * total_loss)
			
		return news

	def predict(self, X, weights):
		outputs = []

		for i in range(len(X)):
			r = self.sigmoid(np.dot(weights, X[i]))
			print(r)
			if  r > 0.5:
				outputs.append(1)
			
			else:
				outputs.append(0)
	
		return outputs

	def evaluate_acc(self, true, predictions):
		correct = 0
		for i in range(len(true)):
			if true[i] == predictions[i]:
				correct += 1
		return correct/len(true)
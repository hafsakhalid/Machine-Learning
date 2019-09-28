import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array


class LogisticRegression:

	def __init__(self, rate, iterations):
		self.rate = rate
		self.iterations = iterations		

	def sigmoid (self, x):
		return 1/(1 + (np.exp(-x)))

	def loss(self, sigmoid, y):
		return (-y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid)).mean()

	def fit (self, X, Y):
		self.weights = np.array([np.zeros(X.shape[1])])

		for _ in range(self.iterations):
			total_loss = 0
			
			for i in range(X.shape[0]):
				delta = self.sigmoid(Y[i] - np.dot(self.weights, X[i]))
				total_loss = np.add(total_loss, X[i] * delta)
			
			self.weights = np.add(self.weights, self.rate * total_loss)
		return self.weights

	def predict(self, X):
		outputs = []
		for x in X:
			print(self.weights)
			r = self.sigmoid(np.dot(self.weights, x))
			
			if  r > 0.5:
				outputs.append(1)
			else:
				outputs.append(0)

		return outputs

with open ('winequality-red.csv', 'r') as f: 
	wines = list(csv.reader(f, delimiter=';'))
	wines = np.array(wines[1:], dtype = np.float)

	quality = np.array(wines[:,11], dtype = np.float) 
	df = pd.DataFrame(quality, columns = ['quality'])
	df.loc[df.quality <=5, 'binary classification'] = 0 
	df.loc[df.quality  >5, 'binary classification'] = 1

	numpy_matrix = df.as_matrix()
	#print(numpy_matrix)
	binaryclassification = (numpy_matrix[:,1])
	y = np.array(binaryclassification)

	X = wines

	rate = 0.0001
	iterations = 10

	model = LogisticRegression(rate, iterations)
	print(model.fit(X, y))
	results = model.predict(wines)
	
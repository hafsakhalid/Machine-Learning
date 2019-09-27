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

	def gradient_descent (self, X, Y, sigmoid):
		return (np.dot(X.T, (Y - sigmoid).T))

	def update (self, weights, rate, gradient):
		return weights - rate * gradient

	def fit (self, X, Y):
		self.weights = np.zeros(X.shape[1])


		for i in range(self.iterations):
			s = self.sigmoid(np.dot(self.weights, X.T))
			gradient = self.gradient_descent(X, Y, s)
			self.weights = self.update(self.weights, self.rate, gradient)
		return self.weights
	

	def predict(self, X):
		outputs = np.zeros(X.shape[1])
		for i in range(outputs.size):
			print(self.weights)
			if self.sigmoid(np.dot(self.weights, X.T)) > 0.5:
				outputs[i] = 1
			else:
				outputs[i] = 0
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
	y = np.array([binaryclassification])

	
	
	X = wines
	
	weights = np.zeros((12, 1), dtype=np.float)
	

	rate = 0.01
	iterations = 100

	model = LogisticRegression(rate, iterations)
	model.fit(wines, y)
	results = model.predict(wines)


   
   
    


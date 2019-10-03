import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array
from numpy import newaxis
import math
import random


class LDA:

	def fit(self, X, y):
		class0 = []
		class1 = []

		for i in range(len(y)):
			if y[i] == 1:
				class1.append(y[i])
			else:
				class0.append(y[i])

		probability0 = len(class0)/len(y)
		probability1 = len(class1)/len(y)

		data1 = np.copy(X[(X[:, -1] > 5)])
		data0 = np.copy(X[(X[:, -1] <= 5)])

		mew0 = np.mean(data0, axis = 0)
		mew1 = np.mean(data1, axis = 0)

		mew00 = mew0[:, newaxis]
		mew11 = mew1[:, newaxis]
		



		avg_v0 = np.zeros((12, 12))
		avg_v1 = np.zeros((12, 12))

		for i in range(len(data0)):
			
			avg_v0 += np.dot((data0[i] - mew00), (data0[i] - mew00).transpose())

		for i in range(len(data1)):
			
			avg_v1 += np.dot((data1[i] - mew11), (data1[i] - mew11).transpose())
			

		avg_sum = avg_v1 + avg_v0

		covariance = avg_sum / (len(X) - 2)
		
		
		values = np.array([probability0, probability1, mew00, mew11, covariance])
		return values

	def predict(self, X, values):
		prob0 = values[0]
		prob1 = values[1]
		avg0 = values[2]
		avg1 = values[3]
		inverse = np.linalg.inv(values[4])

		threshhold = -0.1
		projections = []

		for i in range(len(X)):	
			d_bound = np.log(prob1/prob0) - (1/2) * np.dot(np.dot(avg1.transpose(), inverse), avg1) + (1/2) * np.dot(np.dot(avg0.transpose(), inverse), avg0) + np.dot(np.dot(X[i], inverse), (avg1 - avg0))
			if d_bound > threshhold:
				projections.append(1)
			else:
				projections.append(0)

		return projections


	def evaluate_acc(self, projections, real):
		correct = 0
		for i in range(len(real)):
			if projections[i] == real[i]:
				correct += 1
		return (correct / len(real))






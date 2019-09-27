import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array 

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
	
	#print(binaryclassification)
	#print(binaryclassification.shape)
	#dataset = np.insert(wines, 12, binaryclassification, axis = 1)
	#print(dataset)
	#this is the array of the 1's and 0's 
	#y = []
	#for x in dataset:
		#y.append(x[12])
	#print(y)
	#print(wines)
	#print(y.shape)


#y = binaryclassification
print(y)
print(y.shape)
X = wines
print(X)
w = np.zeros((12, 1), dtype=np.float)
print(w)

def sigmoid (s):
	return 1/1+ (np.exp(-s))

def gradient_descent (X, Y, w):
	sum = 0
	r = Y.size
	for i in range(r):
		#expected gives the value [2,]
		y = np.array([Y[i]])
		print(Y.shape)
		print(y.shape)
		print(y[i])
		x =  np.array([X[i]])
		print(x.shape)
		print(w.shape)
		expected = sigmoid(np.dot(x, w))
		print(expected.shape)
		#expec = np.array([expected])
		print(expected)
		#print(expec.shape)
		d = Y[i] - expected
		#Y[i] gives you an array of 1's and 0's 
		delta = np.array([d])
		print(delta.shape)
		x =  np.array([X[i]])
		print(x.shape)
		for j in X[i]:
			sum += X[i] * delta.T
	return sum

def update (w, a, X, Y):

	for i in range(Y.size):
		w[i] = w[i] + a * gradient_descent(X, Y, w)
	return w

def fit (X, Y, w, iter, a):
	for i in range(iter):
		w = update(w, a, X, Y)
	result = sigmoid (np.dot(w.T,X))

print(fit(wines, y, w, 200, 0.1))




	


   
   
    


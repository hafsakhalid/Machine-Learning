import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 

with open ('winequality-red.csv', 'r') as f: 
	wines = list(csv.reader(f, delimiter=';'))
	wines = np.array(wines[1:], dtype = np.float)
	#print(wines[:,11])
	print(wines.shape)
	#z = np.zeros((1599, 1), dtype=np.int) 
	#np.append(wines, z, axis = 1) 
	#b = np.hstack((wines, np.zeros((wines.shape[0], 1), dtype=wines.dtype)))
	#print(b)
	quality = np.array(wines[:,11], dtype = np.float) 
	#print(quality)
	df = pd.DataFrame(quality, columns = ['quality'])
	#print(df)
	df.loc[df.quality <=5, 'binary classification'] = 0 
	df.loc[df.quality  >5, 'binary classification'] = 1

	#print(df)
	numpy_matrix = df.as_matrix()
	#print(numpy_matrix)
	binaryclassification = (numpy_matrix[:,1])
	#print(binaryclassification)
	dataset = np.insert(wines, 12, binaryclassification, axis = 1)
	print(dataset)
	quality = []
	for x in dataset:
		quality.append(x[12])
	mpl.hist(quality)
	mpl.show()
	
    

class LinearRegression:

	def sigmoid(w, x):
		a = np.dot(w, x)
		return 1/(1 + np.exp(-a))

	def loss(s, y):
		arr = []
		for x in y:
			arr.append(-y * np.log(s) - (1 - y) * np.log(1 - h))
		return np.mean(arr)

	def graddes(w, x, y):
		sum = 0
		for i in range(y.size):
			sum += np.dot(x[i], (y[i] - np.dot(w, x[i])))
		return sum
	
	def update(w, a, g):
		return w - (a * g)





	


   
   
    


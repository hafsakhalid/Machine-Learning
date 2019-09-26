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
	import sys 
	import numpy as np 
	import matplotlib.pyplot as mpl 
	import csv 
	import pandas as pd 

	def __init__(self,x,w):  
		self.x = x;
		self.w = w;

	def sigmoid(w, x):
		a = np.dot(w, x)
		return 1/(1 + np.exp(-a))

	def loss(w, x, y):
		arr = []
		w = 0
		s = sigmoid(w, x)
		for i in len(y):
			w += arr.append(-y[i] * np.log(s) + (1 - y[i]) * np.log(1 - s))
		return w/len(y)

	def descent(w, x, y):
		sum = 0
		for i in range(y.size):
			sum += np.dot(x[i], (y[i] - np.dot(w, x[i])))
		return sum
	
	def update(w, a, g):
		return np.subtract(w(a * g))

	def fit(w, x, y, i, a, sigmoid, loss, descent, update):
		new = x
		for j in i:
			sig = sigmoid(w, new)
			loss = loss(s, y)
			grade = descent(w, new, y)
			new = update(new, a, grade)
		return sigmoid(new, x)

	with open ('winequality-red.csv', 'r') as f: 
		wines = list(csv.reader(f, delimiter=';'))
		wines = np.array(wines[1:], dtype = np.float)

	quality = np.array(wines[:,11], dtype = np.float) 

	df = pd.DataFrame(quality, columns = ['quality'])

	df.loc[df.quality <=5, 'binary classification'] = 0 
	df.loc[df.quality  >5, 'binary classification'] = 1

	
	numpy_matrix = df.as_matrix()
	
	binaryclassification = (numpy_matrix[:,1])
	
	dataset = np.insert(wines, 12, binaryclassification, axis = 1)

	quality = []
	for x in dataset:
		quality.append(x[12])
	mpl.hist(quality)
	mpl.show()
			
	wines = np.array(wines[1:], dtype = np.float)
	data = pd.read_csv("winequality-red.csv")
	quality = np.array(wines[:,11], dtype = np.float)
	df = pd.DataFrame(quality, columns = ['quality'])
	print(df)
	df.loc[df.quality <=5, 'binary classification'] = 0 
	df.loc[df.quality  >5, 'binary classification'] = 1


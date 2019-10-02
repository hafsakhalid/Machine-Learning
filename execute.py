import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array
import math
import random
from log import *
from lda import *




with open ('winequality-red.csv', 'r') as f: 

# #quality = np.array(wines[:,11], dtype = np.float) 
# df = pd.DataFrame(quality, columns = ['quality'])
# df.loc[df.quality <=5, 'binary classification'] = 0 
# df.loc[df.quality  >5, 'binary classification'] = 1
	
	df = pd.read_csv('winequality-red.csv', delimiter=';')
	features = np.array(df.columns)
	feature_arrays = []

	for i in features:
		feature_arrays.append(np.array(df[i]))

	matrix = feature_arrays[len(features) - 1]
	
	if len(features) - 1 == 0:
		matrix = np.column_stack(([1] * len(feature_arrays[0]), matrix))
	else:
		for f in reversed(range(len(features) - 1)):
			matrix = np.column_stack((feature_arrays[f], matrix))
			# if f == 0:
			# 	matrix = np.column_stack(([1] * len(feature_arrays[0]), matrix))

	

	binaryclassification = np.copy((matrix[:,-1]))
	





	for i in range(len(binaryclassification)):
		if binaryclassification[i] > 5.0:
			binaryclassification[i] = 1
		else:
			binaryclassification[i] = 0

	y = binaryclassification
	X = matrix
	k = 5
	def validation(X, k):
		folds = []
		data = X
		fold_size = int(len(data)/k)

		for i in range(k):
			fold = []
			for i in range(fold_size):
				return folds

	folds = validation(X, k)

	rate = 0.002
	iterations = 200
	weights = [1] * len(matrix[0])

	model = LogisticRegression(rate, iterations)
	new_weights = model.fit(X, y, weights)

	results = model.predict(matrix, new_weights)
	print(model.evaluate_acc(y, results))
lda_model = LDA()
logodds = lda_model.fit(X, y)
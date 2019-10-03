import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array
import math
import random
import time
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

	def split(X, k):
		folds = []
		for i in range(k):
			folds.append([])
		data = np.copy(X)

		it = 0
		while len(data) != 0:
			f = it % k
	
			index = random.randint(0, len(data) - 1)

			folds[f].append(data[index])

			data = np.delete(data, index, 0)

			it += 1
		return folds

	rate = 0.0001
	iterations = 10
	weights = [1] * len(matrix[0])

	k = 3
	folds = split(X, k)

	log_results = np.zeros(12)
	log_results = np.reshape(log_results, (12, 1))

	lda_results = np.zeros(5)

	for i in range(k):
		t_set = np.zeros((0, 12))
		v_set = np.zeros((0, 12))
		v_set = np.append(v_set, folds[i], 0)
		
		for j in range(k):
			if j != i:
				t_set = np.append(t_set, folds[j], 0)

		t_set_outputs = np.copy((t_set[:,-1]))

		for i in range(len(t_set_outputs)):
			if t_set_outputs[i] > 5.0:
				t_set_outputs[i] = 1
			else:
				t_set_outputs[i] = 0

		v_set_outputs = np.copy((v_set[:,-1]))
		for i in range(len(v_set_outputs)):
			if v_set_outputs[i] > 5.0:
				v_set_outputs[i] = 1
			else:
				v_set_outputs[i] = 0

		log_model = LogisticRegression(rate, iterations)

		new_weights = log_model.fit(t_set, t_set_outputs, weights)
		new_weights = np.reshape(new_weights, (12, 1))
		

		log_results = np.append(log_results, new_weights, axis=1)
		
		
		lda_model = LDA()
		
		lda_values = lda_model.fit(t_set, t_set_outputs)

		lda_results = np.append(lda_results, lda_values)
		
	
	log_results = np.delete(log_results, 0, axis=1)
	print(log_results)
	log_results_avg = log_results.mean(axis = 1)
	
	lda_results_avg = lda_results.mean()

	


	print(v_set)
	print(v_set_outputs)
	print(log_results_avg)
	log_projections = log_model.predict(v_set, log_results_avg)
	
	
	log_accuracy = log_model.evaluate_acc(v_set_outputs, log_projections)
	
	log_start = time.time()
	log_model = LogisticRegression(rate, iterations)
	new_weights = log_model.fit(t_set, t_set_outputs, weights)
	log_projections = log_model.predict(X, new_weights)
	log_accuracy = log_model.evaluate_acc(t_set_outputs, log_projections)
	log_end = time.time()
	log_time = log_end - log_start 

	print("Logistic Regression Accuracy: {}".format(log_accuracy))
	print("Execution Time: {}".format(log_time))



	lda_model = LDA()
	lda_values = lda_model.fit(t_set, t_set_outputs)
	lda_projections = lda_model.predict(X, lda_values)
	lda_accuracy = lda_model.evaluate_acc(lda_projections, t_set_outputs)

	
	print("LDA Accuracy: {lda_accuracy}")
		



# 	model = LogisticRegression(rate, iterations)
# 	new_weights = model.fit(X, y, weights)

# 	results = model.predict(matrix, new_weights)
# 	print(model.evaluate_acc(y, results))
	
# lda_model = LDA()
# values = lda_model.fit(X, y)
# projections = lda_model.predict(X, values)
# print(lda_model.evaluate_acc(projections, y))
# k_folds = 3










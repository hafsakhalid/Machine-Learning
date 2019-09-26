import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 

with open ('winequality-red.csv', 'r') as f: 
	wines = list(csv.reader(f, delimiter=';'))
	wines = np.array(wines[1:], dtype = np.float)
	quality = []
	for x in wines:
		quality.append(x[11])
	mpl.hist(quality)
	mpl.show()
import sys 
import numpy as np 
import matplotlib.pyplot as mpl 
import csv 
import pandas as pd 
from numpy import array
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
        print(data0)
        print()
        print(data1)

        mew0 = np.mean(data0, axis = 0)
        mew1 = np.mean(data1, axis = 0)

        print(mew0)
        print(mew1)
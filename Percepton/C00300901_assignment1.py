#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:52:58 2019

@author: c00300901
"""

import numpy as np
from sklearn.linear_model import Perceptron
import scipy
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
#1. Generate Sample of Two Classes
x0data = np.transpose(np.array([[np.random.normal(1.0, 1.0) for i in range(5000)], [np.random.normal(-1.0, 1.0) for i in range(5000)]]))
x1data = np.transpose(np.array([[np.random.normal(-1.0, np.sqrt(1.3)) for i in range(5000)], [np.random.normal(1.0, np.sqrt(1.3)) for i in range(5000)]]))
t0vec = -np.ones(5000)
t1vec = np.ones(5000)

#2use	10,000	samples	of	5,000	Class-0	and	5,000	Class-1	to	train	a	Perceptron to	classify	the	samples;

xdata = np.concatenate((x1data, x0data), axis=0)
tvec = np.concatenate((t1vec, t0vec))
shuffle_index = np.random.permutation(10000)
xdata, tvec = xdata[shuffle_index], tvec[shuffle_index]
#Train Perceptron
c_all = Perceptron(tol=1e-3, random_state=0) #Initializing perceptron

#Line_parameters
print("\nFor 1st sample:")
print("Sample Shape: ", xdata.shape)
print("Parameters: ",c_all.fit(xdata, tvec))

print("Coefficient:",c_all.coef_)
print("Intercept:", c_all.intercept_)
print("Iteration: ", c_all.n_iter_)


#To calculcate accuracy, let's make a test data

test_x0data = np.transpose(np.array([[np.random.normal(1.0, 1.0) for i in range(1500)], [np.random.normal(-1.0, 1.0) for i in range(1500)]]))
test_x1data = np.transpose(np.array([[np.random.normal(-1.0, np.sqrt(1.3)) for i in range(1500)], [np.random.normal(1.0, np.sqrt(1.3)) for i in range(1500)]]))
test_t0vec = -np.ones(1500)
test_t1vec = np.ones(1500)

#2use	10,000	samples	of	5,000	Class-0	and	5,000	Class-1	to	train	a	Perceptron to	classify	the	samples;

test_xdata = np.concatenate((test_x1data, test_x0data), axis=0)
test_tvec = np.concatenate((test_t1vec, test_t0vec))
shuffle_index = np.random.permutation(3000)
test_xdata, test_tvec = test_xdata[shuffle_index], test_tvec[shuffle_index]
#Accuracy
print("Synthetic Test Sample Shape:",test_xdata.shape )
print("Accuracy: ",c_all.score(test_xdata, test_tvec))

#3. Use	10,000	samples	of	9,000	Class-0	and	1,000	Class-1	to	train	a	Perceptron to	classify	the	samples;
#report	the	separating	line	parameters;	report	the	accuracy
print("\n \nFor Second Sample:")
#Generating Sample
x0data = np.transpose(np.array([[np.random.normal(1.0, 1.0) for i in range(9000)], [np.random.normal(-1.0, 1.0) for i in range(9000)]]))
x1data = np.transpose(np.array([[np.random.normal(-1.0, np.sqrt(1.3)) for i in range(1000)], [np.random.normal(1.0, np.sqrt(1.3)) for i in range(1000)]]))
t0vec = -np.ones(9000)
t1vec = np.ones(1000)
xdata_new = np.concatenate((x1data, x0data), axis=0)
tvec_new = np.concatenate((t1vec, t0vec))
shuffle_index = np.random.permutation(10000)
xdata_new, tvec_new = xdata_new[shuffle_index], tvec_new[shuffle_index]
#Train Perceptron
c_all_new = Perceptron(tol=1e-3, random_state=0) #Initializing perceptron

#Line_parameters
print("Parameters:  ",c_all_new.fit(xdata_new, tvec_new))

print("Coefficient:",c_all_new.coef_)
print("Intercept:", c_all_new.intercept_)
print("Iteration: ", c_all_new.n_iter_)

#To measure the accuracy, we are going to use our previously generated test data
print("Synthetic Test Sample Shape:",test_xdata.shape )
print("Accuracy:  ",c_all_new.score(test_xdata, test_tvec))


print("\n\nConsidering Cv:")
#4. Repeating step 2 and 3, but considering Cross Validation.
print("Average Accuracy Considering CV=10 for 1st Sample: ",np.average(cross_val_score(c_all, xdata, tvec, cv=10,scoring="accuracy")))
print("Average Accuracy Considering CV=10 for 2nd Sample: ",np.average(cross_val_score(c_all_new, xdata_new, tvec_new, cv=10,scoring="accuracy")))

#5.

print("\n\nPrecision Score:")
prediction_1 = cross_val_predict(c_all, xdata, tvec, cv=10)

print("Precision Score Considering CV=10 for 1st Sample: ", precision_score(tvec, prediction_1)  )

prediction_2= cross_val_predict(c_all_new, xdata_new, tvec_new, cv=10)
print("Precision Score Considering CV=10 for 2nd Sample: ", precision_score(tvec_new, prediction_2) )

print("\n\nRecall Score:")
print("Precision Score Considering CV=10 for 1st Sample: ",recall_score(tvec, prediction_1) )
print("Precision Score Considering CV=10 for 2nd Sample: ",recall_score(tvec, prediction_2) )

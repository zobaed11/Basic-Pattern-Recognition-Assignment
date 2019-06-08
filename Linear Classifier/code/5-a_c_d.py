import numpy as np
import scipy
import sys
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification


#A-PERCEPTRON
print('Perceptron')
X=np.load('train_data.npy')
y=np.load('train_data_label.npy')
c_all = Perceptron(tol=1e-3, random_state=0)
c_all.fit(X, y)

X_test=np.load('test_data.npy')
y_test=np.load('test_data_label.npy')

print("Accuracy:  ",c_all.score(X_test, y_test))







#c
print('.....\nBatch Gradient Descent')
from GradientDescent import BatchGradientDescent
x = np.load('train_data.npy')
y=np.load('train_data_label.npy')
X= np.concatenate( (x.reshape(x.shape[0],2), np.ones(x.shape[0]).reshape(x.shape[0],1)), axis=1  )

cl2= BatchGradientDescent(.1, 500) #Learning factor, number of iterations.
cl2.fit(X,y)


#d
print('.....\nMini Batch Gradient Descent')
from GradientDescent import miniBatchGradientDescent
x = np.load('train_data.npy')
y=np.load('train_data_label.npy')
X= np.concatenate( (x.reshape(x.shape[0],2), np.ones(x.shape[0]).reshape(x.shape[0],1)), axis=1  )
if len(sys.argv) == 1:
    batch_size = X.shape[0]//10         # 10 batches
else:
    batch_size = int(sys.argv[1])

print("Batch Size: ", batch_size)
cl3= miniBatchGradientDescent(.1, 500)
cl3.fit(X,y)





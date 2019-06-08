import numpy as np
import scipy
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


def accuracy():

    testData = np.load('test_data.npy')
    testLabel = np.load('test_data_label.npy')

    tdata = np.concatenate(
        (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).
         reshape(testData.shape[0], 1)), axis=1)

    cnt_right_prediction = 0
    cnt_wrong_prediction = 0

    for idx, point in enumerate(tdata):
        predicted_label = np.matmul(point, w)
        actual_label = testLabel[idx]

        if (actual_label<0 and predicted_label>0) or (
                actual_label>0 and predicted_label<0):
            
            cnt_wrong_prediction+=1
        else:
            
            cnt_right_prediction +=1

    accuracy = cnt_right_prediction/( 
            cnt_right_prediction+cnt_wrong_prediction )
    return accuracy




# input values
x = np.load('train_data.npy')

# observed values
z = np.load('train_data_label.npy')

# data matrix
xdata = np.concatenate((x.reshape(x.shape[0], 2), np.ones(
        x.shape[0]).reshape(x.shape[0], 1)), axis=1)

# pseudo-inverse
xtx = np.matmul(xdata.transpose(), xdata)
xtxinv = np.linalg.inv(xtx)
xinv = np.matmul(xtxinv, xdata.transpose())
w = np.matmul(xinv, z)
print("Weight vector : ", w)

score=accuracy()
print ("psuedo inverse:" ,score)

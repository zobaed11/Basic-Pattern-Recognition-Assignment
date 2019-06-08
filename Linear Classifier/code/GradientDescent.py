import numpy as np
import tensorflow as tf
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


class BatchGradientDescent():
    def __init__(self, eta, n_iter):
        self.eta=eta
        self.n_iter=n_iter
        
    #def calculate_accuracy():




    def fit(self, X, y):
        w = np.random.randn(X.shape[1]) # initialize weights
        N = X.shape[0] #Data Length
        for i in range(self.n_iter):
            error = np.matmul(X, w) - y
            gradient = 2.0 * np.matmul(X.transpose(), error) / N
            w = w - self.eta * gradient
        print("Calculated Weight : ", w)
        testData = np.load('test_data.npy')
        testLabel = np.load('test_data_label.npy')

        tdata = np.concatenate(
            (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)

        cnt_right_prediction = 0
        cnt_wrong_prediction = 0

        for idx, point in enumerate(tdata):
            predicted_label = np.matmul(point, w)
            actual_label = testLabel[idx]

            if (actual_label<0 and predicted_label>0) or (actual_label>0 and predicted_label<0):
                #print("Wrong Prediction: ", actual_label, predicted_label)
                cnt_wrong_prediction = cnt_wrong_prediction + 1
            else:
                #print("Right Prediction: ", actual_label, predicted_label)
                cnt_right_prediction = cnt_right_prediction+1


        accuracy = (cnt_right_prediction)/( cnt_right_prediction+cnt_wrong_prediction )
        print ("Total Wrong Predictions: ", cnt_wrong_prediction )
        print ("Total Right Predictions: ", cnt_right_prediction )
        print ("Accuracy: ", accuracy)


class miniBatchGradientDescent():
    def __init__(self, eta, n_iter):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        def create_mini_batches(X, y, batch_size):
            mini_batches = []
            data = np.hstack((X, y))
            np.random.shuffle(data)
            n_minibatches = data.shape[0] // batch_size
            for i in range(n_minibatches):
                mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
                X_mini = mini_batch[:, :-1]
                y_mini = mini_batch[:, -1] # y is a vector, not a 1-column array
                mini_batches.append((X_mini, y_mini))
                if data.shape[0] % batch_size != 0:
                    mini_batch = data[i * batch_size:data.shape[0]]
                    X_mini = mini_batch[:, :-1]
                    y_mini = mini_batch[:, -1]
                    mini_batches.append((X_mini, y_mini))
            return mini_batches



        def calculate_accuracy():
            testData = np.load('test_data.npy')
            testLabel = np.load('test_data_label.npy')

            tdata = np.concatenate(
                (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)

            cnt_right_prediction = 0
            cnt_wrong_prediction = 0

            for idx, point in enumerate(tdata):
                predicted_label = np.matmul(point, w)
                actual_label = testLabel[idx]

                if (actual_label<0 and predicted_label>0) or (actual_label>0 and predicted_label<0):
                    cnt_wrong_prediction = cnt_wrong_prediction + 1
                else:
                    cnt_right_prediction = cnt_right_prediction+1

            accuracy = cnt_right_prediction/( cnt_right_prediction+cnt_wrong_prediction )
            print ("Total Wrong Predictions: ", cnt_wrong_prediction )
            print ("Total Right Predictions: ", cnt_right_prediction )
            return accuracy


        
        if len(sys.argv) == 1:
            batch_size = X.shape[0]//10         # 10 batches
        else:
            batch_size = int(sys.argv[1])

        w = np.random.randn(X.shape[1]) # initialize weights
        N = X.shape[0] #Data Length
        # find weights iteratively
        for i in range(self.n_iter):
            mini_batches = create_mini_batches(X, y.reshape(y.shape[0], 1), batch_size)

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                error = np.matmul(x_mini, w) - y_mini
                gradient = 2.0 * np.matmul(x_mini.transpose(),error) / batch_size
                w = w - self.eta * gradient

        print(w)

        print("Calculated Weight : ", w)

        accuracy = calculate_accuracy()
        print("Accuracy of the model : ", accuracy)



           


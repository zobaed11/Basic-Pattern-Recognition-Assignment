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
from sklearn.metrics import accuracy_score


def plot_decision_boundary(cl_name,clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')
    plt.savefig(cl_name)



clf = LinearSVC(random_state=0, tol=1e-3)
X=np.load('train_data.npy')
y=np.load('train_data_label.npy')
clf.fit(X, y)
X_test=np.load('test_data.npy')
y_test=np.load('test_data_label.npy')
y_predict=clf.predict(X_test)
sc=accuracy_score(y_test,y_predict)

tot_data=np.load('tot_data.npy')
tot_data_label=np.load("tot_data_label.npy")
print("Linear SVM: ", sc)
#Linear SVM:  0.8333333333333334
plot_decision_boundary("Linear_SVM",clf,tot_data,tot_data_label)

#    sys.exit()



#    sys.exit()

from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
y_predict3=clf2.predict(X_test)
sc2=accuracy_score(y_test,y_predict3)
#sc2=clf2.score(X_test,y_test)
print("Logistic Regression: ", sc2)
plot_decision_boundary("Logistic Regression",clf2,tot_data,tot_data_label)



from sklearn import linear_model

clf1 = linear_model.SGDClassifier( tol=1e-3)
clf1.fit(X, y)
y_predict2=clf1.predict(X_test)
sc1=accuracy_score(y_test,y_predict2)
print("SGD Classifier: ", sc1)
plot_decision_boundary("SGD Classifier",clf1,tot_data,tot_data_label)







c_all = Perceptron(tol=1e-3, random_state=0) #Initializing perceptron
c_all.fit(tot_data, tot_data_label)
y_predict3=c_all.predict(X_test)
sc3=accuracy_score(y_test,y_predict3)
print("Perceptron: ", sc3)
plot_decision_boundary("Perceptron",c_all,tot_data,tot_data_label)



'''
Linear SVM:  0.9666666666666667
Logistic Regression:   0.95
SGD Classifier:  0.8833333333333333
perceptron: .9333

'''
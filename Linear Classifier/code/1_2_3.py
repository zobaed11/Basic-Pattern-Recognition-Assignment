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


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
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
    plt.savefig('perceptron.png')



def plot_data_set(dataMat, labels):
 x,y = dataMat.T
 colors = ['green', 'blue', 'red', 'yellow', 'purple']
 plt.scatter(x, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
 plt.xlim([-6.0,6.0])
 plt.ylim([-6.0,6.0])


tot_data_class_1 = 248
tot_data_class_2 = 50
tot_data_indices=298



# Save datasets as npy file
def save_datasets():
    np.save('tot_data.npy',tot_data)
    np.save('tot_data_label.npy',tot_data_label)
    np.save('train_data.npy', train_data)
    np.save('train_data_label.npy', train_label)
    np.save('test_data.npy', test_data)
    np.save('test_data_label.npy', test_label)

#Class 0 
#class 0 Training Data : Uniform distribution of two variables
    
#flag=False
#if (flag==False):
    
x0_data = np.random.uniform(np.array([-3, -3]), np.array([-.8, 3.2]), np.array([124,2]))
x1_data = np.random.uniform(np.array([-1.2, -3.2]), np.array([2.8, -.8]), np.array([124,2]))

class_0_data = np.concatenate((x1_data, x0_data), axis=0)
#    class_x0_train_data_label = -np.ones(200)
#    class_x1_train_data_label = np.ones(48)

class_0_data_label=-np.ones(248)
#    class_0_train_data_label = np.concatenate((class_x0_train_data_label, class_x1_train_data_label), axis=0)

    
# class 1
# class 1 Train Data : Normal distribution

class_1_data = np.random.normal(np.array([ 1.6, 1.6]), np.array([0.7, 0.7]), np.array([tot_data_class_2,2]))
class_1_data_label = np.ones(50)

#concat two classes
tot_data = np.concatenate((class_0_data, class_1_data), axis=0)
tot_data_label = np.concatenate((class_0_data_label, class_1_data_label), axis=0)

#Upto Here Question 1 

#Shuffle data
shuffle_index = np.random.permutation(tot_data_indices)
tot_data, tot_data_label = tot_data[shuffle_index], tot_data_label[shuffle_index]

# Split into train & test
train_data, test_data,train_label,test_label = train_test_split(tot_data, tot_data_label, test_size=0.20, random_state=50)
#save datasets question 2
save_datasets()


# plot_data_set for question 3
plot_data_set(tot_data, tot_data_label )
plt.savefig('data_distribution_figure.png')





# verify linearly seperable
c_all = Perceptron(tol=1e-3, random_state=0) #Initializing perceptron
c_all.fit(tot_data, tot_data_label)

plot_decision_boundary(c_all,tot_data,tot_data_label)
 







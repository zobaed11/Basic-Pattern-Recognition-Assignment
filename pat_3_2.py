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
import sys
import tensorflow as tf


# function to create a list containing mini-batches
def create_mini_batches(X, z, batch_size):
    mini_batches = []
    data = np.hstack((X, z))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        z_mini = mini_batch[:, -1] # z is a vector, not a 1-column array
        mini_batches.append((X_mini, z_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            z_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, z_mini))
    return mini_batches


xdata=np.load ("concaveData.npy")
tdata= np.load("concaveTarget.npy")

N, p = xdata.shape

vxdata = np.load("testData.npy")
vtdata = np.load("testTarget.npy")


if len(sys.argv) == 1:
    batch_size = xdata.shape[0] // 10 # 10 batches
else:
    batch_size = int(sys.argv[1])
print(batch_size)



# set up network
n_inputs = p
n_hidden1 = 250
n_hidden2 = 125
n_hidden3 = 100
n_hidden4 = 50
n_outputs = 3 # 3 classes
# use placeholder nodes to represent training data and targets
# shape of input is (None, n_inputs)
# assume training data is scaled to [0,1] floating point
# during execution phase, X will be replaced with one training batch at a time
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
t = tf.placeholder(tf.int64, shape=(None), name="t")


# create two hidden layers and output layer
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
activation=tf.nn.relu)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
activation=tf.nn.relu)
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")
    
    
# define cost function to train network
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t,
logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
# define how to train
learning_rate = 0.01
with tf.name_scope("train"):
 optimizer = tf.train.GradientDescentOptimizer(learning_rate)
 training_step = optimizer.minimize(loss)    
 
# how to evaluate model
with tf.name_scope("eval"):
 correct = tf.nn.in_top_k(logits, t, 1)
 accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# initialize
init = tf.global_variables_initializer()
#saver = tf.train.Saver() 

maxx_acc=0.0
n_epochs=200

with tf.Session() as session:
    init.run()
    for epoch in range(n_epochs):
        mini_batches = create_mini_batches(xdata,
                                           tdata.reshape(tdata.shape[0],1), batch_size)
        for mini_batch in mini_batches:
            # do training step first
            X_batch, t_batch = mini_batch
            session.run(training_step, feed_dict={X:X_batch, t:t_batch})
    
     # check accuracies training and test
            acc_train = accuracy.eval(feed_dict={X:X_batch, t:t_batch})
            acc_val = accuracy.eval(feed_dict={X:vxdata, t:vtdata})
            if(maxx_acc<acc_val):
                maxx_acc=acc_val
                epoch_for_maxx=epoch
                max_acc_train=acc_train
#            save_path = saver.save(session,"./model_final.ckpt")    
#        print(epoch, acc_train, acc_val)
        

print("*****DNN Parameters*****")
print("Number of Hidden Layers: ", 4)

print("Neurons in Hidden layer1: ",n_hidden1 )
print("Neurons in Hidden layer2: ",n_hidden2 )
print("Neurons in Hidden layer3: ",n_hidden3 )
print("Neurons in Hidden layer4: ",n_hidden4 )

print("No. of input samples: ", p)
print('DNN Model With default settings, Max accuracy at epoch no. : ',
              epoch_for_maxx, '   Maximum Accuracy of the model : ', maxx_acc)






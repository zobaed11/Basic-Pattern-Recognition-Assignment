import numpy as np
import scipy
import tensorflow as tf
import sys





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



def accuracy():

    testData = np.load('test_data.npy')
    testData = np.concatenate((testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)
    testLabel = np.load('test_data_label.npy')

    # tdata = np.concatenate(
    #     (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)

    cnt_right_prediction = 0
    cnt_wrong_prediction = 0

    for idx, point in enumerate(testData):

        if(np.matmul(point, weightvalues) >=0): 
            predicted_label=1
        else:
            predicted_label=-1
            
        actual_label = testLabel[idx]

        if abs(actual_label-predicted_label) > 0:
            
            cnt_wrong_prediction = cnt_wrong_prediction + 1
        else:
            
            cnt_right_prediction = cnt_right_prediction+1

    accuracy = cnt_right_prediction/( cnt_right_prediction+cnt_wrong_prediction )
    print ("Total Wrong Predictions: ", cnt_wrong_prediction )
    print ("Total Right Predictions: ", cnt_right_prediction )    
    return accuracy


x = np.load("train_data.npy")
zdata = np.load("train_data_label.npy")

xdata= np.concatenate( (x.reshape(x.shape[0],2), np.ones(x.shape[0]).reshape(x.shape[0],1)), axis=1  )
N, p = xdata.shape

X = tf.placeholder(tf.float32, shape=(None, p), name="X") # no length yet
z = tf.placeholder(tf.float32, shape=(None, 1), name="z") # no length yet

# minibatch gradient descent solution
if len(sys.argv) == 1:
    batch_size = xdata.shape[0] // 10 # 10 batches
else:
    batch_size = int(sys.argv[1])
print("Batch size: ", batch_size)

eta = 0.1
n_iterations = 100
Xt = tf.transpose(X)
weights = tf.Variable(tf.random_uniform([p, 1], -1.0, 1.0),
name="weights")
y = tf.matmul(X, weights, name="predictions")
error = y - z
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [weights])[0]

training_step = tf.assign(weights, weights - eta * gradients)

init = tf.global_variables_initializer()
# execution

#ans 6

# make log direction using current time
from datetime import datetime
# current time as string
ctime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, "x")


# create a node that evaluates the mse value and write to binary log string
mse_summary = tf.summary.scalar('mse', mse)
# create logfile writer for summaries
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as session:
    session.run(init)
    for epoch in range(n_iterations):
        mini_batches = create_mini_batches(xdata, zdata.reshape(-1,1), batch_size) 
        for mini_batch in mini_batches:

            x_mini, z_mini = mini_batch
#            session.run(training_step, feed_dict={X:x_mini, z:z_mini.reshape(-1, 1)})
            if epoch % 10 == 0: # write summary
                #session.run(mse_summary,feed_dict={X:x_mini, z:z_mini.reshape(-1,1)})
                summary = session.run(mse_summary,feed_dict={X:x_mini, z:z_mini.reshape(-1,1)})
               
                #sys.exit()
                step = epoch
                file_writer.add_summary(summary, step)
            session.run(training_step, feed_dict={X:x_mini, z:z_mini.reshape(-1,1)})
    weightvalues = weights.eval()
    print(weightvalues)
    
accura = accuracy()
print("Accuracy of the model : ", accura)

file_writer.close()
   

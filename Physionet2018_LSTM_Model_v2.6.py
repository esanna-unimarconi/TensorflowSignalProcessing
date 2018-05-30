""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the Physionet Challenge 2018 database of biometric signals (https://physionet.org/challenge/2018/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Physionet Challenge 2018 Dataset](https://physionet.org/challenge/2018/).
    https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8
@author: Enrico Sanna - Unimarconi
@project: hhttps://github.com/esanna-unimarconi/TensorflowSignalProcessing/
@create-date:16/04/2018

"""
import warnings
#remove future warnings - non funziona
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf
from tensorflow.contrib import rnn
from Physionet2018_LSTM_DataLoading import *
import numpy as np
import pandas as pd
import datetime
import shutil
from logger import Logger as Logger

import os
#remove warnings about system configuration
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    # print ("X size dentro dopo unstack"+str(x))
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Training Parameters
learning_rate = 0.003
training_steps =  100000
test_steps = 5000
batch_size = 1000
display_step = 100

# Network Parameters
num_input =  13  # 28  # number of channels
timesteps = 5  # timesteps: depth (la profondità dei segnali precedenti passata alla lstm
num_hidden = 256  # hidden layer num of features 125
num_classes = 3  # +1: Designates arousal regions; 0: Designates non-arousal regions;-1: Designates regions that will not be scored

print("################################")
print("###### SESSION PARAMETERS ######")
print("################################")
print("learning_rate",learning_rate)
print("training_steps",training_steps)
print("test_steps",test_steps)
print("batch_size",batch_size)
print("display_step",display_step)
print("num_input",num_input)
print("timesteps",timesteps)
print("num_hidden",num_hidden)
print("num_classes",num_classes)
print("################################")
# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


logits = RNN(X, weights, biases)
#print("logits: "+ str(logits))
prediction = tf.nn.softmax(logits)
#print("prediction: "+ str(prediction))
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

dtInizioElaborazione =  datetime.datetime.now()
print(str(datetime.datetime.now()) + " Inizio Elaborazione")

dataLoader = Pysionet2018_LSTM_DataLoading("C:\\PHYSIONET\\")
#dataLoader.next_record_directory()
#dataLoader.next_record_directory()


#preparing data for tensorboard
path="/tmp/Physionet2018_LSTM_2_6"
writer = tf.summary.FileWriter(path)
logger = Logger(path)
#ripulisce la cartella di log
shutil.rmtree(path + "/*", ignore_errors=True)

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        #batch_x = trainX
        #batch_y = trainY
        batch_x, batch_y = dataLoader.train_next_batch(batch_size, timesteps)
        #reformat batch_x in proper shape
        batch_x = rnn_data(batch_x, timesteps, labels=False)
        #print("\n\nX dopo rnn_data" + str(batch_x))
        #print("size X dopo rnn_data" + str(batch_x.size))
        #print("size Y" + str(batch_y))
        #print("size Y" +str(batch_y.size))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            logger.log_histogram('training accuracy', acc, step)
            logger.log_histogram('training loss', loss, step)
            print(str(datetime.datetime.now())+" - Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc)+ \
                  ", record:"+str(dataLoader.getCurrentDirName()) + " sample:"+str(dataLoader.getSampleFrom()))

    print("Optimization Finished!")

    # Calculate accuracy
    #test_data = testX
    #test_label = testY
    #before starting test move to the next record
    #to-do: separate in test folder
    dataLoader.next_record_directory()
    for step in range(1, test_steps + 1):
        test_data, test_label = dataLoader.train_next_batch(batch_size*100, timesteps)
        test_data = rnn_data(test_data, timesteps, labels=False)
        #print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
        acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
        if step % display_step == 0 or step == 1:
            logger.log_histogram('test accuracy', acc, step)
            print(str(datetime.datetime.now()) + " - Step " + str(step)+ ", Testing Accuracy= " + \
                  "{:.3f}".format(acc) + \
                  ", record:" + str(dataLoader.getCurrentDirName()) + " sample:" + str(dataLoader.getSampleFrom()))

dtFineElaborazione = datetime.datetime.now()
elapsedTime= dtFineElaborazione - dtInizioElaborazione
print(str(datetime.datetime.now()) + " Elapsed Time "+ str(elapsedTime))

print("Launching Tensorboard...")
os.system('tensorboard --logdir=' + path)
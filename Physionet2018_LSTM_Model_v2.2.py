""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the Physionet Challenge 2018 database of biometric signals (https://physionet.org/challenge/2018/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Physionet Challenge 2018 Dataset](https://physionet.org/challenge/2018/).
Author: Enrico Sanna - Unimarconi 2
Project: hhttps://github.com/esanna-unimarconi/TensorflowSignalProcessing/
create-date:16/04/2018

"""

import tensorflow as tf
from tensorflow.contrib import rnn

import wfdb
from logger import Logger as Logger
import shutil
import os
import hdf5storage
import numpy as np
import datetime

'''
To extract signals from training dataset
@filename: filepath of the subject
'''
def ExtractSignal(filename):
    # signals_size=5147000
    signals_size = 4770000
    # reading arousal file
    arousal = hdf5storage.loadmat(filename + '-arousal.mat')
    print("dimensione campione")
    print(arousal["data"][0][0][0][0].size)
    # sampling a file from training dataset
    # channel 12 = ECG
    signals, fields = wfdb.rdsamp(filename, sampto=signals_size, channels=[12])
    # signalsArray= signals[:, -1].astype(float)

    # arousal data is the goal o the challenge
    arousalY = np.zeros((125, 2))
    i = 0
    j = 0
    for element in arousal["data"][0][0][0][0]:
        valore = 0 + element
        if (valore >= 0 and j % (4770 * 8) == 0 and j < signals_size):
            arousalY[i][valore] = 1
            i = i + 1
        j = j + 1
        return signals, arousalY


trainX, trainY = ExtractSignal('M:/training/tr03-0005/tr03-0005')
testX, testY = ExtractSignal('M:/training/tr03-0029/tr03-0029')

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 125
display_step = 200

# Network Parameters
num_input = 4770  # 28  # MNIST data input (img shape: 28*28)
timesteps = 8  # timesteps
num_hidden = 125  # hidden layer num of features
num_classes = 2  # +1: Designates arousal regions; 0: Designates non-arousal regions;-1: Designates regions that will not be scored

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

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x = trainX
        # batch_x = arousalData.copy()
        batch_y = trainY
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # batch_x= signals
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            # datetime.date.today()+
            print(" - Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    test_len = 125
    test_data = testX
    test_label = testY
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

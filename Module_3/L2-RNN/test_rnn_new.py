# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:37:02 2018
@author: Franco
Inspired by: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
"""

from __future__ import print_function, division
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

m          = 10000
n_a        = 44
n_y        = 5
n_x        = 24
T          = 10
num_epochs = 1000
batch_size = 1000
num_batches = m/batch_size

#Create random data and output x_data and y_data
y_list = [np.zeros((10000,5)) for i in range(T)]
for y in y_list:
    y[np.arange(m),np.array([np.random.randint(0,5) for j in range(10000)])]=1

y_data = np.stack(y_list).transpose((2,1,0)).astype('int32')
x_data = np.random.randn(n_x,m,T).astype('float32')

#x_data.shape

def create_placeholders():
    x  = tf.placeholder(tf.float32, [n_x,None, T])
    y  = tf.placeholder(tf.int32, [n_y,None, T])
    a0 = tf.placeholder(tf.float32, [n_a,None])
    input_list = tf.unstack(x, axis=2)
    label_list = tf.unstack(y, axis=2)
    return x,y,a0,input_list,label_list

def initialize_parameters(n_x,n_a,n_y):

    Wax = tf.Variable(np.random.rand(n_a, n_x), dtype=tf.float32)
    Waa = tf.Variable(np.random.rand(n_a, n_a), dtype=tf.float32)
    Wya = tf.Variable(np.random.rand(n_y, n_a), dtype=tf.float32)
    ba = tf.Variable(np.random.rand(n_a, 1), dtype=tf.float32)
    by = tf.Variable(np.random.rand(n_y, 1), dtype=tf.float32)

    '''
    Wax = tf.Variable("Wax", [n_a, n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    Waa = tf.Variable("Waa", [n_a, n_a], initializer = tf.zeros_initializer())
    Wya = tf.Variable("Wya", [n_y, n_a], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    ba = tf.Variable("ba", [n_a, 1], initializer = tf.zeros_initializer())
    by = tf.Variable("by", [n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    '''

    parameters = {"Wax": Wax,
                  "Waa": Waa,
                  "Wya": Wya,
                  "ba":  ba,
                  "by":  by}

    return parameters

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = tf.sigmoid(tf.add(tf.add(tf.matmul(Wax, xt), tf.matmul(Waa, a_prev)), ba)) #this is a n_a * m matrix
    yt_pred = tf.add(tf.matmul(Wya, a_next), by)        #this is a n_y * m matrix

    return a_next, yt_pred

def rnn_forward(input_list, a0, parameters):

    a_next = a0
    a_list = []
    logit_list = []
    for input in input_list:  #Iterates period after period
        a_next, yt_pred = rnn_cell_forward(input, a_next, parameters)
        a_list.append(a_next)
        logit_list.append(yt_pred)
        pred_list.append(tf.nn.softmax(yt_pred, 0))

    '''
    a = tf.stack(a_list)
    y_pred = tf.stack(y_list)

    a = tf.transpose(a, [1, 2, 0])
    y_pred = tf.transpose(y_pred, [1, 2, 0])
    '''
    return a_list, logit_list, pred_list

def compute_cost(y_pred, Y):
    print(y_pred.shape)
    print(Y.shape)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

def compute_cost(logit_list, label_list):
    losses = [tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logit_list,label_list)]
    total_loss = tf.reduce_mean(losses)

    return total_loss

def random_mini_batches(x_data, y_data, mini_batch_size,seed):

    np.random.seed(seed)
    m = x_data.shape[1]
    n_y = y_data.shape[0]
    T = y_data.shape[2]

    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = x_data[:, permutation,:]
    shuffled_Y = y_data[:, permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

#        end = 10000 - mini_batch_size * math.floor(10000 / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:,:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def neural_net_rnn(x_data, y_data, n_a, learning_rate, num_epochs, batch_size):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    n_x = x_data.shape[0]
    n_y = y_data.shape[0]
    m = x_data.shape[1]
    T = x_data.shape[2]
    n_a = n_a
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    x,y,a0,input_list,label_list = create_placeholders()

    # Initialize parameters
    parameters = initialize_parameters(n_x,n_a,n_y)
    # Compute forward propagation
#    alist, logit_list= rnn_forward(x, a0, parameters)

    a_next = a0
    a_list = []
    logit_list = []
    pred_list = []
    for input in input_list:  #Iterates period after period
        a_next, yt_pred = rnn_cell_forward(input, a_next, parameters)
        a_list.append(a_next)
        logit_list.append(yt_pred)
        pred_list.append(tf.nn.softmax(yt_pred, 0))

    initial_state = a_next
    # Compute the total loss
    total_loss = compute_cost(logit_list, label_list)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer().minimize(total_loss)  #Leave default

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        loss_list = []

        for epoch_idx in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/batch_size)
            seed = seed + 1
            minibatches = random_mini_batches(x_data, y_data, batch_size, seed)

            i = 0
            for minibatch in minibatches:
                (batchX,batchY) = minibatch
                _initial_state = np.zeros((n_a, batchX.shape[1]))

                _total_loss, _optimizer, _initial_state, _pred_list = sess.run(
                    [total_loss, optimizer, initial_state, pred_list],
                    feed_dict={
                        x:batchX,
                        y:batchY,
                        a0:_initial_state
                    })

                _total_loss = _total_loss / batch_size

                if i == num_minibatches-1:
                    correct_prediction = tf.equal(tf.argmax(_pred_list[-1], axis = 0), tf.argmax(label_list[-1], axis = 0))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("Cost in epoch number %s is ========= %s" % (epoch_idx,_total_loss))
                    print("Accuracy:", accuracy.eval({x: batchX, y: batchY, a0: _initial_state}))

                loss_list.append(_total_loss)
                i += 1

        return parameters, loss_list


parameters, loss_list = neural_net_rnn(x_data, y_data, n_a, 0.01, num_epochs, batch_size)

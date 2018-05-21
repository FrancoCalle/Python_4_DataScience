# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:37:02 2018
@author: Franco
Inspired by: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
"""

from __future__ import print_function, division
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


m          = 10000
n_a        = 25
n_y        = 5
n_x        = 24
T          = 10
num_epochs = 3
batch_size = 1000
num_batches = m/batch_size

#Create random data and output x_data and y_data
y_list = [np.zeros((10000,5)) for i in range(T)]
for y in y_list:
    y[np.arange(m),np.array([np.random.randint(0,5) for j in range(10000)])]=1

y_data = np.stack(y_list).transpose((2,1,0)).astype('int32')
x_data = np.random.randn(n_x,m,T).astype('float32')


def create_placeholders(n_x, n_y, n_a, T):
    x  = tf.placeholder(tf.float32, [n_x,None, T])
    y  = tf.placeholder(tf.int32, [n_y,None, T])
    a0 = tf.placeholder(tf.float32, [n_a,None])
    c0 = tf.placeholder(tf.float32, [n_a, None], name = "C0_HF")

    input_list = tf.unstack(x, axis=2)
    label_list = tf.unstack(y, axis=2)
    return x,y,a0,c0,input_list,label_list

def initialize_parameters(n_x,n_a,n_y):


    Wf = tf.get_variable("Wf", [n_a, n_a + n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    bf = tf.get_variable("bf", [n_a, 1], initializer = tf.zeros_initializer())
    Wi = tf.get_variable("Wi", [n_a, n_a + n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    bi = tf.get_variable("bi", [n_a, 1], initializer = tf.zeros_initializer())
    Wc = tf.get_variable("Wc", [n_a, n_a + n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    bc = tf.get_variable("bc", [n_a, 1], initializer = tf.zeros_initializer())
    Wo = tf.get_variable("Wo", [n_a, n_a + n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    bo = tf.get_variable("bo", [n_a, 1], initializer = tf.zeros_initializer())
    Wy = tf.get_variable("Wy", [n_y, n_a], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    by = tf.get_variable("by", [n_y, 1], initializer = tf.zeros_initializer())



    parameters = {  'Wf' : Wf,
                    'bf' : bf,
                    'Wi' : Wi,
                    'bi' : bi,
                    'Wc' : Wc,
                    'bc' : bc,
                    'Wo' : Wo,
                    'bo' : bo,
                    'Wy' : Wy,
                    'by' : by}

    return parameters

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    concat = tf.concat([a_prev,xt], 0)   #Concatenate a and x to multiply with the gate parameters


    ft = tf.sigmoid(tf.add(tf.matmul(Wf, concat) , bf))
    it = tf.sigmoid(tf.add(tf.matmul(Wi, concat) , bi))
    cct = tf.tanh(tf.add(tf.matmul(Wc, concat), bc))
    c_next = tf.add(tf.multiply(ft, c_prev), tf.multiply(it, cct))
    ot = tf.nn.sigmoid(tf.add(tf.matmul(Wo, concat), bo))
    a_next = tf.multiply(ot, tf.tanh(c_next))

    yt_pred = tf.add(tf.matmul(Wy, a_next) , by)    # Compute the Logits

    return a_next, c_next, yt_pred

def compute_cost(logit_list, label_list):

    losses = [tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(logits), labels = tf.transpose(labels)) for logits, labels in zip(logit_list,label_list)]
    total_loss = tf.reduce_mean(losses)

    return total_loss

def random_mini_batches(x_data, y_data, mini_batch_size,seed):

    np.random.seed(seed)
    m = x_data.shape[1]
    n_x = x_data.shape[0]
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
    x,y,a0,c0,input_list,label_list = create_placeholders(n_x, n_y, n_a, T)

    # Initialize parameters
    parameters = initialize_parameters(n_x,n_a,n_y)

    # Compute forward propagation
    a_next = a0
    c_next = c0
    a_list = []
    logit_list = []
    pred_list = []
    for input in input_list:  #Iterates period after period
        a_next, c_next, yt_pred = lstm_cell_forward(input, a_next, c_next, parameters)
#        a_list.append(a_next)
        logit_list.append(yt_pred)
        pred_list.append(tf.nn.softmax(yt_pred, 0))

    initial_state_a = a_next
    initial_state_c = c_next
    # Compute the total loss
    total_loss = compute_cost(logit_list, label_list)
    # Backpropagation: Define the tensorflow optimizer. Use an AdagradOptimizer.
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        loss_list = []
        accuracy_list_0 = []
        accuracy_list_T = []
        for epoch_idx in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/batch_size)
            seed = seed + 1
            minibatches = random_mini_batches(x_data, y_data, batch_size, seed)

            i = 0
            for minibatch in minibatches:
                (batchX,batchY) = minibatch
                _initial_state_a = np.random.randn(n_a, batchX.shape[1])
                _initial_state_c = np.random.randn(n_a, batchX.shape[1])

                _total_loss, _optimizer, _initial_state_a, _initial_state_c, _pred_list = sess.run(
                    [total_loss, optimizer, initial_state_a, initial_state_c, pred_list],
                    feed_dict={
                        x:batchX,
                        y:batchY,
                        a0:_initial_state_a,
                        c0:_initial_state_c
                    })

                _total_loss = _total_loss / num_minibatches

                #Prediction in time 0:
                correct_prediction_0 = tf.equal(tf.argmax(_pred_list[0], axis = 0), tf.argmax(label_list[0], axis = 0))
                accuracy_0 = tf.reduce_mean(tf.cast(correct_prediction_0, "float"))
                accuracy_0 = accuracy_0.eval({x: batchX, y: batchY, a0: _initial_state_a, c0: _initial_state_c})
                accuracy_list_0.append(accuracy_0)

                #Prediction in time T:
                correct_prediction_T = tf.equal(tf.argmax(_pred_list[-1], axis = 0), tf.argmax(label_list[-1], axis = 0))
                accuracy_T = tf.reduce_mean(tf.cast(correct_prediction_T, "float"))
                accuracy_T = accuracy_T.eval({x: batchX, y: batchY, a0: _initial_state_a, c0: _initial_state_c})
                accuracy_list_T.append(accuracy_T)

#                if i == num_minibatches-1:
                if i % 5 == 0:
                    print("Cost in epoch number %s is ========= %s" % (epoch_idx,_total_loss))
                    print("Accuracy 0:", accuracy_0)
                    print("Accuracy T:", accuracy_T)

#                print("Accuracy:", accuracy.eval({x: batchX, y: batchY, a0: _initial_state}))


                loss_list.append(_total_loss)
                i += 1


        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        #Print Results:
        (pd.Series(loss_list)).plot(style = 'b-')
        (pd.Series(accuracy_list_T)).plot(style = 'm-',secondary_y = True)
        (pd.Series(accuracy_list_0)).plot(style = 'g-',secondary_y = True).legend(['Time=T','Time=0','Loss'])

        return parameters, loss_list


parameters, loss_list = neural_net_rnn(x_data, y_data, n_a, 0.01, num_epochs, batch_size)

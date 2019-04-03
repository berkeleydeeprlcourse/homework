#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:12:49 2019

@author: zhaoxuanzhu
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import gym

def tf_reset():
    try: 
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()

def create_model(n_observation,n_action,regularization):
    # create inputs
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_observation])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_action])
    # create variables
    W0 = tf.get_variable(name='W0', shape=[n_observation, 128], initializer=tf.contrib.layers.xavier_initializer())
    wd = tf.nn.l2_loss(W0)*regularization
    tf.add_to_collection("weight_decay",wd)
    W1 = tf.get_variable(name='W1', shape=[128, n_action], initializer=tf.contrib.layers.xavier_initializer())
    wd = tf.nn.l2_loss(W1)*regularization
    tf.add_to_collection("weight_decay",wd)    
    
    b0 = tf.get_variable(name='b0', shape=[128], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[n_action], initializer=tf.constant_initializer(0.))
    
    weights = [W0, W1]
    biases = [b0, b1]
    activations = [tf.nn.relu, None]
    
    # create computation graph
    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer
    
    return input_ph, output_ph, output_pred

def tf_training(actions,observations,n_steps,sess,input_ph, output_ph, output_pred):
    
#    sess = tf.Session()
    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
    weight_decay_loss = tf.reduce_mean(tf.get_collection("weight_decay"))
    total_loss = mse+weight_decay_loss
    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(total_loss)
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()
    
    # run training
    for training_step in range(n_steps):
        # get a random subset of the training data
        input_batch = observations
        output_batch = actions
        
        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, total_loss], feed_dict={input_ph: input_batch, output_ph: output_batch})
        
        # print the mse every so often
        if training_step % 10 == 0:
            print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
            save_path = saver.save(sess, 'trainingresults/hopper.ckpt')
            
    print("Model saved in path: %s" % save_path)
    summary_writer = tf.summary.FileWriter("/tmp/logs", sess.graph)
    summary_writer.close()

    
#def main(envname,Train_Restore):
envname = "Hopper-v2"
Train_Restore = 1

with open(os.path.join('expert_data', envname + '.pkl'), 'rb') as f:
    data_from_expert = pickle.load(f)

actions_expert = data_from_expert['actions']
observations_expert = np.float32(data_from_expert['observations'])
actions_expert_processed = np.divide((actions_expert-np.mean(actions_expert,0)),np.std(actions_expert,0))
observations_expert_processed = np.divide((observations_expert-np.mean(observations_expert,0)),np.std(observations_expert,0))

sess = tf_reset() 
n_action = actions_expert.shape[1]
n_observation = observations_expert.shape[1]

input_ph, output_ph, output_pred = create_model(n_observation,n_action,0.001)

if Train_Restore ==0:    
    tf_training(actions_expert_processed,observations_expert_processed,2000,sess,input_ph, output_ph, output_pred)
elif Train_Restore == 1:
    saver = tf.train.Saver()
    saver.restore(sess, "trainingresults/hopper.ckpt")

#env = gym.make(envname)
#max_steps = env.spec.timestep_limit
#observations = np.zeros(shape=(1,11))
#actions = np.zeros(shape=(1,3))
#
#obs = env.reset()
#obs = np.reshape(obs,(1,11))
#done = False
#totalr = 0.
#steps = 0
#while not done:
#    action = sess.run(output_pred, feed_dict={input_ph:obs})
#    observations = np.append(observations,obs,axis = 0)
#    actions = np.append(actions,action,axis = 0)
#    obs, r, done, _ = env.step(action)
#    obs = np.reshape(obs,(1,11))
#    totalr += r
#    steps += 1
#    env.render()
#    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
#    if steps >= max_steps:
#        break
#print(totalr)
#env.render()
#env.close()

prediction = sess.run(output_pred, feed_dict={input_ph: np.reshape(observations_expert_processed[0,:],(1,n_observation))})
print(prediction)

#main('Hopper-v2',0)
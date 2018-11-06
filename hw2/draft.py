import tensorflow as tf
import numpy as np

# x = tf.placeholder("float", [None, 3])
# y = x * 2

# batch index
# https://stackoverflow.com/questions/51052203/tensorflow-indexing-according-to-batch-position
ob_dim = 4
logits=tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
actions=tf.placeholder(shape=[None], name="ac", dtype=tf.int32)

# prepare row indices
row_indices = tf.range(tf.size(actions))
action_indices = tf.stack([row_indices, actions], axis = 1)
prob = tf.gather_nd(logits, action_indices)


with tf.Session() as session:
	logit_val = [[0.1, 0.3, 0.4, 0.2], [0.3,0.2,0.1,0.4], [0.15,0.5,0.25,0.1]]
	action_val = [2, 0, 3]
	print(session.run(prob, feed_dict={logits: logit_val, actions: action_val}))


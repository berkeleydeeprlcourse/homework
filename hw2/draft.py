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



def sum_of_rewards(re_n, reward_to_go, gamma):
    # YOUR_CODE_HERE
    sum_of_path_lengths = sum([len(re) for re in re_n])
    q_n = np.zeros(sum_of_path_lengths)
    counter = 0
    if reward_to_go:
        for re in re_n:
        	start = counter
        	for i, r in enumerate(re[::-1]):
        		index = start + len(re) - 1 - i
        		print(index)
        		if i == 0:
        			q_n[index] = r
        		else:
        			q_n[index] = r +  gamma* q_n[index+1]
        		counter+=1
    else:
    	for re in re_n:
        	for i, r in enumerate(re):
        		if i == 0:
        			q_n[counter] = r
        		else:
        			q_n[counter] = gamma**(i) * r + q_n[counter-1]
        		counter+=1
    return q_n

actual_false = (sum_of_rewards([[0, 2, 3], [24], [0, 2, 3], [0, 6]], False, 0.9))
expected_false = ([0, 0 + 2 * 0.9, 0 + 2*0.9 + 3*0.9*0.9, 24, 0, 0+2*0.9, 0+2*0.9+0.81*3, 0, 6*0.9])
assert(actual_false, expected_false)

actual_true = (sum_of_rewards([[0, 2, 3], [24], [0, 2, 3], [0, 6]], True, 0.9))
expected_true = ([2*0.9+3*0.9**2, 2+3*0.9, 3, 24, 2*0.9+3*0.9**2, 2+3*0.9, 3, 6*0.9, 6])
assert(actual_true, expected_true)

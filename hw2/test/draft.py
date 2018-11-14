import tensorflow as tf
import numpy as np
import itertools

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


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
        print(cumulative)
    
    # mean = np.mean(discounted_episode_rewards)
    # std = np.std(discounted_episode_rewards)
    # discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards


def sum_of_rewards(re_n, reward_to_go, gamma):
    # YOUR_CODE_HERE
    sum_of_path_lengths = sum([len(re) for re in re_n])
    
    counter = 0
    cumulative = 0.0
    if reward_to_go:
        q_n = list(itertools.chain(*[discount_and_normalize_rewards(re, gamma) for re in re_n]))
        # for re in re_n:
        #     cumulative = 0
        #     lim = len(re) - 1
        #     start = counter
        #     for i, r in enumerate(re[::-1]):
        #         index = start + lim - i
        #         cumulative = cumulative * gamma + r
        #         q_n[index] = cumulative
        #         counter+=1
    else:
        q_n = np.zeros_like(range(sum_of_path_lengths))
        for re in re_n:
            cumulative = 0
            for i, r in enumerate(re):
                cumulative = cumulative * gamma + r
                q_n[counter] = cumulative
                counter+=1
    return q_n

# actual_false = (sum_of_rewards([[0, 2, 3], [24], [0, 2, 3], [0, 6]], False, 0.9))
# expected_false = ([0, 0 + 2 * 0.9, 0 + 2*0.9 + 3*0.9*0.9, 24, 0, 0+2*0.9, 0+2*0.9+0.81*3, 0, 6*0.9])

# actual_true = (sum_of_rewards([[0, 2, 3], [24], [0, 2, 3], [0, 6]], True, 0.9))
# expected_true = ([2*0.9+3*0.9**2, 2+3*0.9, 3, 24, 2*0.9+3*0.9**2, 2+3*0.9, 3, 6*0.9, 6])


value = discount_and_normalize_rewards([0, 1, 3, 4, 5, 6], 0.95)
print("A", value)

# value = sum_of_rewards([[0, 1, 3, 4, 5, 6]], False, 0.95)
# print("total reward", value)

value = sum_of_rewards([[0, 1, 3, 4, 5, 6]], True, 0.95)
print("reward-to-go", value)

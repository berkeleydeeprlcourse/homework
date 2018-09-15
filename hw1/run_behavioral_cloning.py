import argparse
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import multiprocessing
import numpy as np

import tf_util


def load_data(task, test_percent_split=20):
    data = pickle.load(open('expert_data/{}.pkl'.format(task), 'rb'))
    obvs, acts = data['observations'], data['actions']
    x_train, x_test, y_train, y_test = train_test_split(obvs, acts, test_size=test_percent_split / 100.)
    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}


def create_model(hidden_units, in_size):
    obv_ph = tf.placeholder(dtype=tf.float32, shape=[None, in_size], name="obv_ph")
    act_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1, hidden_units[-1]], name="act_ph")
    net = obv_ph
    for i, num_hidden in enumerate(hidden_units):
        W = tf.get_variable(name='W{}'.format(i), shape=[in_size, num_hidden],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b{}'.format(i), shape=[num_hidden], initializer=tf.constant_initializer(0.))
        net = tf.matmul(net, W, name="MatMul_layer{}".format(i)) + b
        if i != len(hidden_units) - 1:
            net = tf.nn.relu(net)
        in_size = num_hidden
    return obv_ph, act_ph, tf.expand_dims(net, axis=1)


def train_model(envname, x_train, x_test, y_train, y_test, hidden_units, in_size, num_epochs, num_cores=1,
                batch_size=32):
    save_name = 'behavior_cloning_data/{}/model.ckpt'.format(envname)
    sess = tf_util.make_session(num_cores)
    obv_ph, act_ph, net = create_model(hidden_units, in_size)
    mse = tf.reduce_mean(0.5 * tf.square(net - act_ph))
    opt = tf.train.AdamOptimizer().minimize(mse)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    iter = 0
    for epoch in range(num_epochs):
        for training_step in range(x_train.shape[0] // batch_size):
            input_batch = x_train[training_step * batch_size: training_step * batch_size + batch_size]
            output_batch = y_train[training_step * batch_size: training_step * batch_size + batch_size]
            _, mse_run = sess.run([opt, mse], feed_dict={obv_ph: input_batch, act_ph: output_batch})
            iter += 1

            if training_step % 1000 == 0:
                print('EPOCH #{}, iter {} mse: {}'.format(epoch, training_step, mse_run))
                saver.save(sess, save_name)
    sess.close()
    return save_name


def eval_policy(envname, checkpoint, hidden_units, in_size, max_timesteps=None, num_rollouts=None, render=False):
    tf.reset_default_graph()
    sess = tf.Session()
    obv_ph, act_ph, net = create_model(hidden_units, in_size)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(net, feed_dict={obv_ph: obs[None, :]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


def main(envname: str, test_percent_split: int = 20, render: bool = False, num_rollouts: int = 20, num_epochs: int = 10,
         batch_size: int = 128):
    data = load_data(envname, test_percent_split)
    model = train_model(envname, **data, num_epochs=num_epochs, num_cores=multiprocessing.cpu_count(),
                        hidden_units=[80, 60, 40, np.prod(data['y_train'].shape[1:])],
                        in_size=data['x_train'].shape[1], batch_size=batch_size)
    eval_policy(envname, model, num_rollouts=num_rollouts,
                hidden_units=[80, 60, 40, np.prod(data['y_train'].shape[1:])],
                in_size=data['x_train'].shape[1], render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--test_percent_split', type=int, default=0,
                        help='Percent of training data to use for test/val (not implemented)')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(**vars(args))

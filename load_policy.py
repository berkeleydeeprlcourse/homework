"""
Code to load an expert policy and save out data for behavioral cloning.
Example usage:
    python load_policy.py humanoid.pkl Humanoid-v1 --render \
            --output_file expert_data.pkl --num_rollouts 20
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util

def load_policy(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user
    obs_dim = policy_params['obsnorm']['Standardizer']['mean_1_D'].shape[1]
    action_dim = policy_params['out']['AffineLayer']['b'].shape[1]

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        output_bo = tf.matmul(curr_activations_bd, W) + b
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo)
    policy_fn = tf_util.function([obs_bo], a_ba)
    return policy_fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--output_file', type=str, default='expert_data.pkl')
    args = parser.parse_args()

    print('loading')
    build_policy, obs_dim, action_dim = load_policy(args.input)
    print('loaded', obs_dim, action_dim)

    obs_bo = tf.placeholder(tf.float32, [None, None])
    print('building')
    a_ba = build_policy(obs_bo)
    print('built')

    policy_fn = tf_util.function([obs_bo], a_ba)

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'):
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations), 'actions': np.array(actions)}
        with open(args.output_file, 'wb') as f:
            data = pickle.dump(expert_data, f)

if __name__ == '__main__':
    main()

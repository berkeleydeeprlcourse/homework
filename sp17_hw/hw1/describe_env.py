import argparse
import glob
import re
from os import path

import gym


def describe_env(envname, verbose=False):
    print('loading {} environment'.format(envname))
    env = gym.make(envname)

    action_space = env.action_space
    obs_space = env.observation_space

    try:
        action_space.high
        action_is_box = True
    except AttributeError:
        action_is_box = False

    try:
        obs_space.high
        obs_is_box = True
    except AttributeError:
        obs_is_box = False

    if verbose:
        print('action space: ')
        print(env.action_space)
        if action_is_box:
            print('lower bounds: {}'.format(env.action_space.low))
            print('upper bounds: {}'.format(env.action_space.high))
            print()

        print('observation space:')
        print(env.observation_space)

        if obs_is_box:
            print('lower bounds: {}'.format(env.observation_space.low))
            print('upper bounds: {}'.format(env.observation_space.high))
            print()

    else:
        if action_is_box:
            print('Action space is {} dimensional, '.format(action_space.shape[0]), end='')
        else:
            print('Action space is discrete with {} possible values, '.format(action_space.shape[0]), end='')
        if obs_is_box:
            print('observation space is {} dimensional'.format(obs_space.shape[0]))
        else:
            print('observation space is discrete with {} possible values, '.format(obs_space.shape[0]), end='')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', type=str, default=None,
                        help='If specified, describe a single environment in detail. If not, give brief '
                             'description of all expert environments.')
    args = parser.parse_args()

    if args.envname is None:
        fnames = glob.glob('./experts/*')

        for fname in fnames:
            envname = re.findall('(.+).pkl', path.basename(fname))[0]
            describe_env(envname)
            print()


    else:
        describe_env(args.envname, verbose=True)

if __name__ == '__main__':
    main()

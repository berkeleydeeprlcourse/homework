NUM_ROLLOUTS_TO_TEST = [5, 10, 15, 20, 25]

import argparse

import matplotlib.pyplot as plt
import numpy as np

from run_expert import main as run_expert_main
from run_policy import main as run_policy_main
from behavioral_cloning import main as behavioral_cloning_main

def run_expert_rollouts(envname, num_rollouts):
    expert_data_filename = 'expert_data/expert_data_{}'.format(envname)
    expert_policy_file = 'experts/{}.pkl'.format(envname)
    expert_returns = run_expert_main(expert_policy_file, envname, num_rollouts, expert_data_filename=expert_data_filename)
    print('expert return mean')
    print(np.mean(expert_returns))
    print('expert return stdev')
    print(np.std(expert_returns))
    return expert_returns

def train_behavioral_cloning_model(envname):
    expert_data_filename = 'expert_data/expert_data_{}'.format(envname)
    model_filepath = 'models/{}'.format(envname)
    training_accuracies = behavioral_cloning_main(expert_data_filename, envname, model_filepath)
    print('training accuracy start and end')
    print(training_accuracies[0])
    print(training_accuracies[-1])
    return training_accuracies

def run_trained_policy(envname):
    model_filepath = 'models/{}'.format(envname)
    num_rollouts = 20
    policy_rewards = run_policy_main(model_filepath, envname, num_rollouts)
    print('cloned policy return mean')
    print(np.mean(policy_rewards))
    print('cloned policy return stdev')
    print(np.std(policy_rewards))
    return policy_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()

    envname = args.envname

    num_experiments = 5

    policy_reward_mean = []
    policy_reward_stdev = []
    for num_rollouts in NUM_ROLLOUTS_TO_TEST:
        # do average of num_experiments
        expert_rewards = []
        policy_rewards = []
        for i in range(num_experiments):
            expert = run_expert_rollouts(envname, num_rollouts)
            train_behavioral_cloning_model(envname)
            policy = run_trained_policy(envname)
            expert_rewards.append(expert)
            policy_rewards.extend(policy)
        print('number of expert rollouts {}'.format(num_rollouts))
        print('mean of trained policy reward {}'.format(np.mean(policy_rewards)))
        print('stdev of trained policy reward {}'.format(np.std(policy_rewards)))
        policy_reward_mean.append(np.mean(policy_rewards))
        policy_reward_stdev.append(np.std(policy_rewards))

    plt.plot(NUM_ROLLOUTS_TO_TEST, policy_reward_mean)
    plt.ylabel('mean policy reward over {} iterations of the experiments'.format(num_experiments))
    plt.xlabel('number of expert policy rollouts')
    plt.savefig('expert_rollouts_versus_trained_policy_mean_reward' + envname)

    plt.plot(NUM_ROLLOUTS_TO_TEST, policy_reward_stdev)
    plt.ylabel('stdev of policy reward over {} iterations of the experiments'.format(num_experiments))
    plt.xlabel('number of expert policy rollouts')
    plt.savefig('expert_rollouts_versus_trained_policy_reward_stdev' + envname)

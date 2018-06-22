import pickle
import numpy as np
from sklearn.utils import shuffle

# TODO: pass in logger


class Data(object):
    def __init__(self, data_file, train_ratio, val_ratio):
        data = pickle.load(open(data_file, "rb"))

        self.expert_mean_return=data['mean_return']
        self.expert_std_return=data['std_return']

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        obs = np.stack(data['observations'], axis=0)
        actions = np.squeeze(np.stack(data['actions'], axis=0))
        assert len(obs) == len(actions), "obs and action mismatch!"

        obs, actions = shuffle(obs, actions, random_state=0)

        self.num_observations = obs.shape[1]
        self.num_actions = actions.shape[1]

        print("Splitting dataset...")
        self.train, self.val, self.test = self.split(obs, actions)

        self.print_stat(self.train, "Training")
        self.print_stat(self.val, "Validation")
        self.print_stat(self.test, "Test")

        obs_mean = np.mean(self.train["observations"], axis=0)
        obs_std = np.std(self.train["observations"], axis=0)

        print("Normalizing observations...")
        self.pre_proc(self.train, obs_mean, obs_std)
        self.pre_proc(self.val, obs_mean, obs_std)
        self.pre_proc(self.test, obs_mean, obs_std)

    def split(self, obs, actions):
        """Split the dataset into train, val, and test"""     
        n_total = len(obs)
        n_train, n_val = int(n_total * self.train_ratio), int(n_total * self.val_ratio)

        train_data = {"observations": obs[:n_train], "actions": actions[:n_train]}
        val_data = {"observations": obs[n_train:n_train + n_val], "actions": actions[n_train:n_train + n_val]}
        test_data = {"observations": obs[n_train + n_val:], "actions": actions[n_train + n_val:]}

        return train_data, val_data, test_data

    def get_small_dataset(self, num_data=100):
        """Return a subset of the training data"""
        obs, actions = self.train["observations"], self.train["actions"]
        idx = np.random.choice(np.arange(len(obs)), size=num_data, replace=False)
        small_data = {"observations": obs[idx], "actions": actions[idx]}
        return small_data

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """Batch generator for a dataset"""
        num_data = len(data["observations"])
        num_batch_per_epoch = int((num_data-1) / batch_size) + 1

        for epoch in range(num_epochs):
            obs, actions = data["observations"], data["actions"]
            if shuffle:
                idx = np.random.permutation(np.arange(num_data))
                obs = obs[idx]
                actions = actions[idx]
            for i in range(num_batch_per_epoch):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_data)
                yield obs[start_idx:end_idx], actions[start_idx:end_idx]

    @staticmethod
    def print_stat(data, title):
        obs, actions = data["observations"], data["actions"]
        print("%s Observations %s, mean: %s" % (title, str(obs.shape), str(np.mean(obs, axis=0))))
        print("%s Actions %s, mean: %s" % (title, str(actions.shape), str(np.mean(actions, axis=0))))

    @staticmethod
    def pre_proc(data, mean, std):
        """Normalize observations"""
        obs = data["observations"]
        data["observations"] = (obs - mean) / (std + 1e-6)  # See load_policy.py

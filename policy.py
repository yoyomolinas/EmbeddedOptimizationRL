import random
import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, epsilon_scheduler = None, action_space = None):
        """

        :param epsilon_scheduler: epsilon_scheduler object define in schedulers.py
        :param action_space: gym.spaces.Space instance. Could be Discrete(2).
        """
        if epsilon_scheduler is not None:
            self.epsilon_scheduler = epsilon_scheduler
        else:
            from parameterSchedulers import linearScheduler
            self.epsilon_scheduler = linearScheduler(0.5,c=1,k=0.004)

        self.param = self.epsilon_scheduler.param
        self.action_space = action_space
        self.name = 'epsilon-greedy'

    def update(self):
        self.epsilon_scheduler.update()
        self.param = self.epsilon_scheduler.param

    def sample(self, qV = None):
        """

        :param qV: An array of q values for each action
        :return: action
        """
        if random.random() < self.epsilon_scheduler.param or (qV is None):
            return self.action_space.sample()
        else:
            qV = np.squeeze(qV)
            return np.argmax(qV)


class GreedyPolicy:
    def __init__(self, action_space=None):
        self.action_space = action_space
        self.name = 'greedy'
        self.param = 0.
    def update(self):
        pass


    def sample(self, qV = None):
        """

        :param qV: An array of q values for each action
        :return: action
        """
        assert qV is not None
        qV = np.squeeze(qV)
        return np.argmax(qV)

class AlwaysDoPolicy:
    def __init__(self, action_space=None):
        self.action_space = action_space
        self.name = 'always-do'
        self.param = 0.
    def update(self):
        pass

    def sample(self, qV = None):
        """

        :param qV: An array of q values for each action
        :return: action
        """
        return 1


class RandomPolicy:
    def __init__(self, p = 0.5, action_space=None):
        self.action_space = action_space
        from parameterSchedulers import linearScheduler
        self.epsilon_scheduler = linearScheduler(p, c=1, k=0)
        self.name = 'random'
        self.param = 1.

    def update(self):
        self.epsilon_scheduler.update()

    def sample(self, qV = None):
        """

        :param qV: An array of q values for each action
        :return: action
        """
        return self.action_space.sample()

class PositiveRewardPolicy:
    def __init__(self, action_space = None):
        self.action_space = action_space
        self.name = 'positive-reward'
        self.param = 0.
    def sample(self, reward = None):
        if reward > 0:
            return 1
        elif reward < 0:
            return 0
        else:
            return self.action_space.sample()

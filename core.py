import keras
import numpy as np
import gym
from metrics import Metric

"""
Base algorithm class
"""

class BaseAlgo:
    def __init__(self, env = None, mdp = None, policy = None):
        self.env = env
        self.mdp = mdp
        self.policy = policy
        self.model = self.get_model()
        # create metrics
        self.metrics = {'train': {}, 'test': {}, 'replay':{}}
        for mode in self.metrics.keys():
            self.metrics[mode]['accumulated_reward'] = Metric(name = 'accumlated reward')
            self.metrics[mode]['reward'] = Metric(name = 'reward')
            self.metrics[mode]['pred_reward'] = Metric(name = 'reward')
            self.metrics[mode]['q_vals'] = Metric(name = 'q values')
            self.metrics[mode]['loss'] = Metric(name = 'loss')
            self.metrics[mode]['epsilon'] = Metric("epsilon")
            self.metrics[mode]['action'] = Metric("action")

    def get_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(128,
                                     input_shape=(self.mdp.state_class.num_dims,),
                                      kernel_initializer='random_uniform'))
        model.add(keras.layers.Dense(64,
                                     kernel_initializer='random_uniform'))
        model.add(keras.layers.Dense(2,
                                     kernel_initializer='random_uniform'))

        return model

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def export(self, load_path, save_path):
        self.model = self.get_model()
        self.load_weights(load_path)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        self.model.save(save_path)

"""
Abstract state class
"""
class AbstractState(gym.Space):
    def __init__(self, encoding = None, prev_state=None):
        """
        :param encoding: Encoding object defined in data_utils.py
        :param prev_state: State object representing previous state. None if no previous state
        """
        self.num_dims = None # Number of dimensions in vector form of this state
        self.encoding = encoding
        self.feature = encoding.feature
        self.fid = encoding.frame_id
        self.action = None
        self.done_flag = False
        t, l, b, r = self.encoding.box
        self.resolution = max(b - t, r - l)

        if prev_state is None:
            self.confs = {'age': np.expand_dims(np.zeros(len(self.encoding.age_conf), ) + 0.000001, 0),
                          'gender': np.expand_dims(np.zeros(len(self.encoding.gender_conf), ) + 0.000001, 0)}
            self.n = len(self.confs['age'])
            self.probs = {'age': self.confs['age'],
                          'gender': self.confs['gender']}
            self.entropy = {'age': np.sum(self.probs['age'] * np.log2(1 / self.probs['age'])),
                            'gender': np.sum(self.probs['gender'] * np.log2(1 / self.probs['gender']))}

            self.entropy_change = 0  # negative if current entropy > prev entropy

        else:
            self.confs = prev_state.confs
            self.n = prev_state.n
            self.probs = prev_state.probs
            self.entropy = prev_state.entropy
            self.entropy_change = 0  # negative if current entropy > prev entropy

    def confidence(self, key='gender'):
        """
        Helper method used in reward function
        :param key: 'gender' or 'age'
        :return: maximum of latest recorded confidence score of given key
        """
        assert self.action is not None, "self.action should be updated but in this state"
        if self.action == 0:
            return 0
        else:
            return max(self.confs[key][0])

    def prob(self, key='gender'):
        """
        :param key: 'gender' or 'age'
        :return: the probability of class with maximum likelihood for given key
        """
        return np.max(self.probs[key])

    def vectorize(self):
        """
        Concatenate information representing this state
        :return: a vector of shape (1, self.num_dims)
        """
        raise NotImplementedError


    def update(self, action):
        """
        Update self.action. Must be called after doing an action in order to store
        confidences, probs, entropy information in this state.
        :param action: 0 or 1
        """
        assert not self.done_flag, "this state has been updated before"
        self.action = action
        self.done_flag = True
        # Update self.confs, self.n, self.probs, self.entropy if action is taken
        if self.action == 1:
            confs = {'age': np.expand_dims(self.encoding.age_conf, 0),
                     'gender': np.expand_dims(self.encoding.gender_conf, 0)}
            self.confs['age'] = np.concatenate([confs['age'], self.confs['age']], axis=0)
            self.confs['gender'] = np.concatenate([confs['gender'], self.confs['gender']], axis=0)
            self.n = len(self.confs['age'])
            self.probs = {'age': np.mean(self.confs['age'], axis=0), 'gender': np.mean(self.confs['gender'], axis=0)}
            cur_entropy = {'age': np.sum(self.probs['age'] * np.log2(1 / self.probs['age'])),
                            'gender': np.sum(self.probs['gender'] * np.log2(1 / self.probs['gender']))}
            # Compute entropy_change as self.entropy - cur_entropy
            self.entropy_change = 0.5 * (self.entropy['age'] - cur_entropy['age']) + \
                                  0.5 * (self.entropy['gender'] - cur_entropy['gender'])
            self.entropy = cur_entropy

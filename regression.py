# from tensorflow import keras
import keras
from keras import backend as K
import numpy as np
from tqdm import tqdm
import time
from metrics import Metric
from core import BaseAlgo
from policy import AlwaysDoPolicy

BOOTSRAP_CONSTANT = np.pi

def bootstrap_loss(y_true, y_pred):
    y_true = K.tf.squeeze(y_true)
    y_pred = K.tf.squeeze(y_pred)
    idx = K.tf.equal(y_true, BOOTSRAP_CONSTANT)
    y_true_boot = K.cast(np.zeros(2, ), K.floatx())
    y_true_boot += y_pred * K.cast(idx, K.floatx())
    y_true_boot += y_true * K.cast(K.tf.logical_not(idx), K.floatx())
    return K.mean(K.square(y_pred - y_true_boot))


class RegressionAlgo(BaseAlgo):
    def __init__(self, mdp=None, env=None, policy=None):
        super().__init__(mdp=mdp, env=env, policy=policy)
        self.name = 'regression'
        self.loss_func = bootstrap_loss
        self.model.compile(loss=self.loss_func, optimizer=keras.optimizers.Adam())
        self.memory = []

    def collect_data(self, n = 1, mode = 'train'):
        self.set_mode(mode)
        self.mdp.reset()
        self.env.reset(video_path=self.mdp.video_path)
        for i in tqdm(range(n * self.mdp.data.size())):
            # choose action
            action = self.policy.sample()

            prev_state_vector = self.mdp.state.vectorize()

            # take action
            prev_state, action, state = self.mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action)

            # append to memmory
            self.memory.append((prev_state_vector, action, reward))

            if self.mdp.is_terminal:
                self.mdp.reset()

    def fit(self, epochs=1):
        X = []
        Y = []
        # Prepare x and Y
        # reset metrics
        for name, metric in self.metrics['train'].items():
            metric.reset()

        for prev_state_vector, action, reward in self.memory:
            X.append(prev_state_vector)
            if action == 1:
                target = [BOOTSRAP_CONSTANT, reward]
            else:
                target = [reward, BOOTSRAP_CONSTANT]
            Y.append(target)
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        self.model.fit(X, Y, batch_size=8, epochs=epochs, verbose=1, shuffle=True)

    def test(self, policy, visualize = False):
        self.set_mode('test')
        if visualize:
            self.env.set_mode('human')
        else:
            self.env.set_mode('test')

        self.mdp.reset()
        self.env.reset(video_path=self.mdp.video_path)

        # reset metrics
        for name, metric in self.metrics['test'].items():
            metric.reset()

        for i in tqdm(range(self.mdp.data.size())):

            pred_reward = np.squeeze(self.model.predict(self.mdp.state.vectorize()))

            # choose action -- 0 if reward < 0 else 1
            action = policy.sample(pred_reward)

            # take action
            prev_state, action, state = self.mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action)

            self.metrics['test']['accumulated_reward'].add_append(reward)
            self.metrics['test']['pred_reward'].append(pred_reward)
            self.metrics['test']['reward'].append(reward)
            self.metrics['test']['action'].append(action)

            self.env.render(fid=prev_state.fid, box=prev_state.encoding.box,
                            action=action,
                            accumulated_reward=self.metrics['test']['accumulated_reward'].sum,
                            reward=reward,
                            k = prev_state.k)

            if self.mdp.is_terminal:
                self.mdp.reset()
                self.env.reset(video_path=self.mdp.video_path)

        print("Testing accumulated reward :", self.metrics['test']['accumulated_reward'].history[-1])
        time.sleep(.2)

    def set_mode(self, mode='train'):
        assert mode in ['test', 'train']
        # If mode is human, need lot of data to filter out videos that are hard to load, so set mode to train
        self.mdp.set_mode(mode)

    def reset_memory(self):
        self.memory = []

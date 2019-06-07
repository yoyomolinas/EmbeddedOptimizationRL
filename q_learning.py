import keras
import numpy as np
import random
from tqdm import tqdm
import time
from utils import progressBar
from metrics import Metric
from core import BaseAlgo


class QLearningAlgo(BaseAlgo):
    def __init__(self, env = None, mdp = None, policy = None, discount=0.9, max_iter = 1e3):
        super().__init__(env = env, mdp = mdp, policy = policy)
        self.name = 'q-learning'
        self.discount = discount
        self.num_epsiodes = 0

        self.loss_func = 'mse'
        self.model.compile(loss=self.loss_func, optimizer=keras.optimizers.Adam())

        self.memory = []

    def set_mode(self, mode='train'):
        assert mode in ['test', 'train']
        self.mdp.set_mode(mode)

    def remember_one(self, prev_state, action, state, reward, is_terminal):
        self.memory.append((prev_state, action, state, reward, is_terminal))

    def replay(self, batch_size = 16, epochs = 1):
        X = []
        Y = []
        # print("Running replay...")
        c = 0
        for e in range(epochs):
            print('Epoch %i/%i'%(e+1,epochs))
            time.sleep(.2)
            self.memory = random.sample(self.memory, k = len(self.memory))
            for m in tqdm(self.memory):
                c += 1
                if len(X) == batch_size:
                    X = np.concatenate([X], axis = 0)
                    Y = np.concatenate([Y], axis = 0)
                    result = self.model.fit(X, Y, epochs = 1, verbose = 0)
                    # update metrics
                    self.metrics['replay']['loss'].append(result.history['loss'])

                    X, Y = [], []
                prev_state_vector, action, next_state_vector, reward, is_terminal = m
                q_vals = self.model.predict(prev_state_vector)
                if is_terminal:
                    # End of episode
                    q_next = 0
                else:
                    q_next = np.max(self.model.predict(next_state_vector))
                q_vals[0][action] = reward + q_next * self.discount
                X.append(np.squeeze(prev_state_vector))
                target = q_vals
                Y.append(np.squeeze(target))
                self.metrics['replay']['q_vals'].append(q_vals)
                self.metrics['replay']['accumulated_reward'].add_append(reward)
                self.metrics['replay']['reward'].append(reward)

                # verbose
                # verbose_info = " -- epoch : %i -- accumulated reward : %.3f"%\
                #                (e, self.metrics['replay']['accumulated_reward_metric'].sum)
                # progressBar("Replay Progress" + verbose_info, c, epochs * len(self.memory))



    def fit(self, epochs = 1, remember = True):
        """
        Fit model
        :return: accumulated rewards per episode and loss list
        """
        self.set_mode('train')
        self.mdp.reset()
        verbose_title = "Fit Progress"
        # reset metrics
        for name, metric in self.metrics['test'].items():
            metric.reset()

        for i in tqdm(range(epochs * self.mdp.data.size())):
            # compute q values
            q_vals = self.model.predict(self.mdp.state.vectorize())

            # choose action
            action = self.policy.sample(qV=q_vals)

            prev_state_vector = self.mdp.state.vectorize()

            # take action
            prev_state, action, state = self.mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action)

            # append to memory to replay later
            if remember:
                self.remember_one(prev_state_vector, action, self.mdp.state.vectorize(), reward, self.mdp.is_terminal)

            # update epsilon and set q_next
            if self.mdp.is_terminal:
                # End of episode
                self.policy.update()
                self.num_epsiodes += 1
                q_next = 0
            else:
                q_next = np.max(self.model.predict(state.vectorize()))

            q_target = reward + self.discount * q_next
            target = q_vals
            target[0][action] = q_target

            result = self.model.fit(prev_state_vector, target, epochs = 1, verbose = 0)

            # render environemnt
            self.env.render(fid = self.mdp.fid,
                            box = self.mdp.state.encoding.box,
                            action=action,
                            reward = reward)

            # update metrics
            self.metrics['train']['q_vals'].append(q_vals)
            self.metrics['train']['accumulated_reward'].add_append(reward)
            self.metrics['train']['reward'].append(reward)
            self.metrics['train']['loss'].append(result.history['loss'])
            self.metrics['train']['action'].append(action)
            self.metrics['train']['epsilon'].append(self.policy.param)

            # verbose
            # verbose_info = " -- epoch : %i -- episode : %i -- accumulated reward : %.3f"%\
            #                (self.mdp.itercycle, self.num_epsiodes, self.metrics['train']['accumulated_reward_metric'].sum)
            # progressBar(verbose_title + verbose_info, i, epochs * self.mdp.data.size())

            # reset mdp and env if terminal state
            if self.mdp.is_terminal:
                self.mdp.reset()
        print("Training accumulated reward :", self.metrics['train']['accumulated_reward'].history[-1])
        time.sleep(.2)

    def test(self, policy = None, visualize = False):
        self.set_mode('test')
        # reset metrics
        for name, metric in self.metrics['test'].items():
            metric.reset()

        if visualize:
            self.env.set_mode('human')
        else:
            self.env.set_mode('test')
        self.mdp.reset()
        self.env.reset(video_path = self.mdp.video_path)

        verbose_title = "Test Progress"

        for i in tqdm(range(self.mdp.data.size())):
            # compute q values
            q_vals = self.model.predict(self.mdp.state.vectorize())

            # sampe action
            action = policy.sample(qV=q_vals)

            # do action
            prev_state, action, state = self.mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action)

            # update metrics
            self.metrics['test']['q_vals'].append(q_vals)
            self.metrics['test']['accumulated_reward'].add_append(reward)
            self.metrics['test']['reward'].append(reward)
            self.metrics['test']['action'].append(action)

            # verbose
            # verbose_info = " -- video %s -- accumulated reward : %.3f" % \
            #                (mdp.video_path, self.metrics['test']['accumulated_reward_metric'].sum)
            # progressBar(verbose_title + verbose_info, i, mdp.data.size())

            self.env.render(fid = prev_state.fid, box = prev_state.encoding.box,
                            action = action,
                            accumulated_reward=self.metrics['test']['accumulated_reward'].sum)

            if self.mdp.is_terminal:
                self.mdp.reset()
                self.env.reset(video_path=self.mdp.video_path)


        print("Testing accumulated reward :", self.metrics['test']['accumulated_reward'].history[-1])
        time.sleep(.2)
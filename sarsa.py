import keras
from tqdm import tqdm
import numpy as np
import random
from .utils import progressBar
from .basealgo import BaseAlgo

"""
SARSA
"""

class SARSA(BaseAlgo):
    def __init__(self, env = None, mdp = None, policy = None, discount=0.9, max_iter = 1e3):
        super().__init__(env = env, mdp = mdp, policy = policy)
        self.discount = discount
        self.num_epsiodes = 0

    def set_mode(self, mode='train'):
        assert mode in ['train', 'test']
        self.mdp.set_mode(mode)

    def remember_one(self, prev_state, action, state, reward, is_terminal):
        self.memory.append((prev_state, action, state, reward, is_terminal))

    def replay(self, batch_size = 16, epochs = 1):
        X = []
        Y = []
        # print("Running replay...")
        c = 0
        for e in range(epochs):
            self.memory = random.sample(self.memory, k = len(self.memory))
            for m in self.memory:
                c += 1
                if len(X) == batch_size:
                    X = np.concatenate([X], axis = 0)
                    Y = np.concatenate([Y], axis = 0)
                    result = self.model.fit(X, Y, epochs = 1, verbose = 0)
                    # update metrics
                    self.metrics['replay']['loss_metric'].append(result.history['loss'])

                    X, Y = [], []
                prev_state, action, state, reward, is_terminal = m
                q_vals = self.model.predict(prev_state.vectorize())
                # TODO

    def fit(self, epochs = 1, remember = True):
        """
        Fit model
        :return: accumulated rewards per episode and loss list
        """

        self.set_mode('train')
        self.env.set_mode('train')

        self.mdp.reset()
        self.env.reset(video_path = self.mdp.video_path)


        verbose_title = "Fit Progress"
        for i in tqdm(range(self.mdp.data.size())):
            # Cover initial state scenario
            if self.mdp.state is None:
                # choose action
                action = self.policy.sample()
            else:
                # compute q values
                q_vals = self.model.predict(self.mdp.state.vectorize())

                # choose action
                action = self.policy.sample(qV=q_vals)

            # take action
            prev_state, action, state = self.mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action, state)

            # reset if necessary
            if self.mdp.is_terminal:
                self.num_epsiodes += 1
                self.reset()

            # append to memory to replay later
            if remember:
                self.remember_one(prev_state, action, state, reward, self.mdp.is_terminal)

            # update epsilon and set q_next
            if self.mdp.is_terminal:
                self.policy.update()
                self.num_epsiodes += 1


            else:
                q_next = self.model.predict(state.vectorize())
                action_next = self.policy.sample(qV=q_next)
                q_target = reward + self.discount * np.squeeze(q_next)[action_next]
                target = q_vals
                target[0][action] = q_target

            result = self.model.fit(prev_state.vectorize(), target, epochs = 1, verbose = 0)

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

            # verbose
            verbose_info = " -- epoch : %i -- episode : %i -- accumulated reward : %.3f"%\
                           (self.mdp.itercycle, self.num_epsiodes, self.metrics['train']['accumulated_reward_metric'].sum)
            progressBar(verbose_title + verbose_info, i, epochs * self.mdp.data.size())

            # reset mdp and env if terminal state
            if self.mdp.is_terminal:
                self.mdp.reset()
                self.env.reset(video_path=self.mdp.video_path)

    def test(self, mode = 'human', mdp = None, policy = None):
        if mdp is None:
            mdp = self.mdp
        if policy is None:
            policy = self.policy

        # reset metrics
        for name, metric in self.metrics['test'].items():
            metric.reset()

        self.env.mode = mode
        mdp.reset()
        if mode == 'human':
            print(mdp.video_path)
            while mdp.video_path in ['../../Vids/test/ekrem_imamoglu.mp4', '../../Vids/test/new_york_1993.mp4']:
                mdp.reset()

        mdp.itercycle = 0
        self.env.reset(video_path = mdp.video_path)
        self.num_epsiodes = 0

        verbose_title = "Test Progress"

        i = 0

        while mdp.itercycle == 0:
            i += 1

            # Cover initial state scenario
            if self.mdp.state is None:
                # choose action
                action = 1 # self.policy.sample(qV = None)

                # take action
                self.mdp.doAction(action)

                # reset if necessary
                if self.mdp.is_terminal:
                    self.num_epsiodes += 1
                    self.mdp.reset()
                    self.env.reset(video_path=self.mdp.video_path)

                continue

            # compute q values
            q_vals = self.model.predict(mdp.state.vectorize())

            # sampe action
            action = policy.sample(qV=q_vals)

            # do action
            prev_state, action, state = mdp.doAction(action)

            # compute reward
            reward = self.env.compute_reward(prev_state, action, state)

            self.env.render(fid = mdp.fid, box = mdp.state.encoding.box,
                            action=action,
                            accumulated_reward=self.metrics['test']['accumulated_reward_metric'].sum)

            # update metrics
            self.metrics['test']['q_vals_metric'].append(q_vals)
            self.metrics['test']['accumulated_reward_metric'].add_append(reward)
            self.metrics['test']['reward_metric'].append(reward)

            # verbose
            verbose_info = " -- video %s -- accumulated reward : %.3f" % \
                           (mdp.video_path, self.metrics['test']['accumulated_reward_metric'].sum)
            progressBar(verbose_title + verbose_info, i, mdp.data.size())

            if mdp.is_terminal:
                mdp.reset()
                if mode == 'human':
                    print(mdp.video_path)
                    while mdp.video_path in ['../../Vids/test/ekrem_imamoglu.mp4', '../../Vids/test/new_york_1993.mp4']:
                        mdp.reset()

                self.env.reset(video_path=mdp.video_path)

    def reset(self):
        assert self.mdp.is_terminal
        self.mdp.reset()
        self.env.reset(video_path=self.mdp.video_path)


import data_utils
from mdp import MDP
from rewards import reward_func_linear  # Call it with stats to initialize
from env import Env
from q_learning import QLearningAlgo
from policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy

data = data_utils.Data(n=15)
mdp = MDP(data=data)
reward_func = reward_func_linear(data.statistics, verbose=False)
env = Env(reward_func=reward_func, mode = 'human')
# policy = EpsilonGreedyPolicy(action_space = mdp.action_space)
policy = RandomPolicy(action_space = mdp.action_space)
test_policy = GreedyPolicy(action_space = mdp.action_space)
algo = QLearningAlgo(env = env, mdp= mdp, policy = policy, discount = 0.2)

algo.set_mode('train')
algo.fit(mode = 'train', epochs = 4, remember=True)

algo.set_mode('test')
algo.test(mode = 'test', policy = test_policy)

algo.replay(batch_size=16, epochs = 8)

algo.set_mode('test')
algo.test(mode = 'test', policy = test_policy)

# algo.test(mode = 'human', policy = test_policy)

import numpy as np
import matplotlib.pyplot as plt
def plot():
    for mode in ['train','test', 'replay']:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(mode)
        axes = axes.flatten()

        axes[0].set_title("Loss")
        axes[0].plot(algo.metrics[mode]['loss_metric'].history)

        axes[1].set_title("Accumulated reward over time")
        axes[1].plot(algo.metrics[mode]['accumulated_reward_metric'].history)

        axes[2].set_title("Reward over time")
        axes[2].scatter(range(len(algo.metrics[mode]['reward_metric'].history)), algo.metrics[mode]['reward_metric'].history, s = 0.3)

        q_vals = np.squeeze(algo.metrics[mode]['q_vals_metric'].history)
        axes[3].set_title("Q_values of actions wrt. each other")
        axes[3].set_ylabel("action 1")
        axes[3].set_xlabel("action 0")
        axes[3].scatter(q_vals[:, 0], q_vals[:, 1], s = 0.1)

    plt.show()

plot()
q_vals = np.squeeze(algo.metrics['test']['q_vals_metric'].history)
print('percentage of greedy actions')
sum(np.argmax(q_vals, axis = 1)) / len(q_vals)
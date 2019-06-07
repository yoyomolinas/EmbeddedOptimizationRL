import gym
import numpy as np
from core import AbstractState

class SimpleState(AbstractState):
    num_dims = 1030
    k_max = 2.
    def __init__(self, encoding = None, prev_state = None):
        super().__init__(encoding = encoding, prev_state = prev_state)
        # Determine coeff based on this: self.k should be 0.5 if 5 actions are taken.
        # coeff**5 ~ 0.5
        self.coeff = 0.9
        if prev_state is None:
            # Play with initial value of k to alter reward trajectory. See reward function
            self.k = self.k_max
        else:
            self.k = prev_state.k

    def update(self, action):
        super(SimpleState, self).update(action)
        if self.action == 1:
            self.k = self.k * self.coeff

    def vectorize(self):
        """
        Concatenate information representing this state
        :return: a vector of shape (1, self.num_dims)

        """
        assert not self.done_flag, "cannot vectorize a state that has been updated before, contains information about rewards"
        vec1 = self.feature
        vec2 = np.array([self.resolution,
                         self.entropy['gender'],
                         self.entropy['age'],
                         self.prob(key='gender'),
                         self.prob(key='age'),
                         self.k,
        ])
        vec_concat = np.concatenate([vec1, vec2])
        vec_final = np.expand_dims(vec_concat, axis = 0)
        return vec_final

class MDP:
    action_space = gym.spaces.Discrete(2)  # Action space, 0 or 1
    def __init__(self, data = None, state_class = SimpleState):
        """

        :param dataiter: DataIterator object defined in data_utils.py
        """
        self.data = data
        self.dataiter = iter(self.data)
        self.itercycle = 0
        self.state_class = state_class # One of the state classes defined above
        self.episode = None # A generator defined in dataiter.episode_iterator
        self.prev_state = None # An instance of state class
        self.state = None # An instance of state class
        self.fid = None # A number representing current frame id
        self.video_path = None # Video path from where current episode was obtained
        self.is_terminal = None # Set True if episode returns done = True, else False

    def doAction(self, action):
        assert self.action_space.contains(action), "Action should be 0 or 1."
        assert not self.is_terminal, "In terminal state, reset mdp"

        self.state.update(action)
        self.prev_state = self.state

        encoding, done = next(self.episode)
        if done:
            self.is_terminal = True
        self.fid = encoding.frame_id
        self.state = self.state_class(encoding, prev_state=self.prev_state)
        return self.prev_state, action, self.state

    def reset(self):
        try:
            episode_iterator, self.video_path  = next(self.dataiter)
        except StopIteration as e:
            self.dataiter = iter(self.dataiter)
            self.itercycle += 1
            episode_iterator, self.video_path = next(self.dataiter)

        self.episode = iter(episode_iterator)
        encoding, done = next(self.episode)
        self.state = self.state_class(encoding=encoding, prev_state=None)
        if done:
            return self.reset()
        self.is_terminal = False
        self.prev_state = None

    def set_mode(self, mode):
        if mode in ['train', 'test']:
            self.data.mode = mode
            self.dataiter = iter(self.data)

if __name__ == '__main__':
    # Test
    import data_utils
    data = data_utils.Data(small = True)
    mdp = MDP(data = data)
    mdp.reset()

    for i in range(1000):
        mdp.doAction(1)
        if mdp.is_terminal:
            mdp.reset()




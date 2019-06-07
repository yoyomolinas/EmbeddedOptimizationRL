import copy
import gym
import cv2
import sys
import video

"""
Some global variables. Useful in visualizing frames

"""
age_groups = [[0, 8], [8, 15], [15, 20], [20, 25], [25, 32], [32, 35], [35, 38], [38, 43], [43, 48], [48, 53], [53, 60],
              [60, 200]]

genders = ['Female', 'Male']


class Env(gym.Env):
    """
    A class that inherits from the gym.Env.
    :param: mdp: an instance of mdp.MDP class
    :param: reward_func: an instance of reward functions defined in rewards.py
    :param: policy: instance of policy defined in policies.py
    :param: mode: if mode is human env is going to render and visualize the environment
    """

    def __init__(self, mdp = None, reward_func = None, mode='human'):
        # self.mdp = mdp
        self.compute_reward = reward_func # Callable
        self.cap = None  # VideoCapture object
        self.buffer = None  # Buffer of frames in current video
        self.window_name = 'frame'  # Used in rendering
        self.window_size = (1280, 720) # (1280, 720)
        self.mode = mode

    def draw(self, frame =  None, box = None, **to_print):
        """
        Draw visualize track, action, age, gender in given frame
        :param frame: an array of pixels
        :param to_print: items be print on frame
        :return: edited frame
        """
        frame = copy.deepcopy(frame)
        t, l, b, r = list(map(int, box))
        cv2.rectangle(frame, (l, t), (r, b), (255, 100, 100), 2)

        text = ""
        for key, val in to_print.items():
            if key == 'gender':
                key == 'GENDER'
                val = genders[val]
            elif key == 'age':
                key = 'AGE'
                val = age_groups[val]
            elif key == 'action':
                key = 'ACTION'
                if val == 1:
                    frame[t:b, l:r][:, :, 0] = 20
                val = 'predict' if val == 1 else 'not predict'
            elif key == 'reward':
                key = 'REWARD'
                val = round(val, 3)

            text += " %s: %s"%(str(key), str(val))

        cv2.putText(frame, text, (0, 700), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 100, 220), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, self.window_size)

        return frame

    def render(self, fid, **to_print):
        """
        Render current frame and visualize track, action, reward etc.
        :param to_print: items in this dictionary will be print on screen.
        """
        if self.mode == 'human':
            frame = self.draw(self.buffer[int(fid) - 1] , **to_print)

            cv2.imshow(self.window_name, frame)

            k = cv2.waitKey(30)
            if k == 27:
                sys.exit(0)

    # def step(self, action):
    #     """
    #
    #     :param action: 0 or 1
    #     :return: previous_state, reward, state where states are instances of slef.mdp.state_class
    #     """
    #     assert not self.mdp.is_terminal, "Reset env before querying next step"
    #     prev_state, action, state = self.mdp.doAction(action)
    #     reward = self.compute_reward(prev_state, action, state)
    #     return prev_state, reward, state

    def reset(self, video_path = None):
        """
        1. reset self.mdp
        2. if mode is 'human' load next video into buffer
        """

        if self.mode == 'human':
            assert video_path is not None
            if self.cap is not None and self.cap.pipeline == video_path:
                return

            elif self.cap is not None and video_path != self.cap.pipeline:
                self.cap.close()

            self.cap = video.VideoCapture(video_path, shape=(720, 1280))

            # Read all frames into self.buffer
            self.buffer = []
            while True:
                # if len(self.buffer) % 1000 == 0 and len(self.buffer) != 0:
                #     print("Loading %ith frame into buffer" % len(self.buffer))
                try:
                    self.buffer.append(self.cap.read())  # Read frame
                except BufferError as e:
                    break
            # print("Video in buffer:", self.cap.pipeline, "\t# of frames in buffer:", len(self.buffer))

    def set_mode(self, mode):
        self.mode = mode

if __name__ == '__main__':
    # Test
    import data_utils
    from mdp import MDP
    from rewards import reward_func_linear # Call it with stats to initialize
    data = data_utils.Data(n = 2)
    mdp = MDP(dataiter = iter(data))
    reward_func = reward_func_linear(data.statistics, verbose = False)
    env = Env(mdp = mdp, reward_func=reward_func)
    while True:
        if env.mdp.is_terminal:
            env.reset()

        _, reward, _ = env.step(1)
        env.render(gender = env.mdp.state.encoding.gender, age = env.mdp.state.encoding.age, reward = reward, action = 1)


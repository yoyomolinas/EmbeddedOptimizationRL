import numpy as np


"""
Get reward function as lambda
"""
def reward_func_linear(stats, init_cost = 1., verbose=False):
    mean_conf_gender, mean_conf_age, mean_entropy_change, dev_conf_gender, dev_conf_age, dev_entropy_change = stats
    if verbose:
        print("Maximum confidences in GENDER classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_gender, dev_conf_gender))
        print("Maximum confidences in AGE classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_age, dev_conf_age))
        print("Entropy change has mean %.4f and standard deviation %.4f" % (
            mean_entropy_change, dev_entropy_change))

    # Params: action -> 0 or 1, states -> State objects
    def func(prev_state, action):
        c_g = prev_state.confidence('gender')
        c_a = prev_state.confidence('age')

        cost = init_cost
        reward_conf = .5 * (((c_g - mean_conf_gender) / dev_conf_gender) + 1) + \
                      .5 * (((c_a - mean_conf_age) / dev_conf_age) + 1)

        alpha = .5
        # Compute entropy change.
        # Negative if current state's entropy is higher than previous state's.
        # Reward decreasing entropy change
        ec = prev_state.entropy_change

        reward_entropy = .5 * (((ec - mean_entropy_change) / dev_entropy_change) + 1)

        # Clip max min values
        if reward_entropy < 0:
            reward_entropy = 0
        elif reward_entropy > 2:
            reward_entropy = 2

        reward = action * ((alpha * reward_conf + (1 - alpha) * reward_entropy) - cost)
        return reward

    return func

def reward_func_decay(stats, init_cost = .8, verbose=False):
    mean_conf_gender, mean_conf_age, mean_entropy_change, dev_conf_gender, dev_conf_age, dev_entropy_change = stats
    if verbose:
        print("Maximum confidences in GENDER classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_gender, dev_conf_gender))
        print("Maximum confidences in AGE classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_age, dev_conf_age))
        print("Entropy change has mean %.4f and standard deviation %.4f" % (
            mean_entropy_change, dev_entropy_change))

    # Params: action -> 0 or 1, states -> State objects
    def func(prev_state, action):
        c_g = prev_state.confidence('gender')
        c_a = prev_state.confidence('age')

        cost = init_cost
        reward_conf = .5 * (((c_g - mean_conf_gender) / dev_conf_gender) + 1) + \
                      .5 * (((c_a - mean_conf_age) / dev_conf_age) + 1)


        cost += cost + (prev_state.k_max - prev_state.k)
        alpha = .5
        # Compute entropy change.
        # Negative if current state's entropy is higher than previous state's.
        # Reward decreasing entropy change
        ec = prev_state.entropy_change
        reward_entropy = .5 * (((ec - mean_entropy_change) / dev_entropy_change) + 1)

        # Clip max min values
        if reward_entropy < 0:
            reward_entropy = 0
        elif reward_entropy > 2:
            reward_entropy = 2

        reward = action * ((alpha * reward_conf + (1 - alpha) * reward_entropy) - cost)
        return reward

    return func

def reward_func_simple_decay(stats, init_cost = 0.8, verbose=False):
    """
    Decay with k, no entropy
    :param stats:
    :param verbose:
    :return:
    """
    mean_conf_gender, mean_conf_age, mean_entropy_change, dev_conf_gender, dev_conf_age, dev_entropy_change = stats
    if verbose:
        print("Maximum confidences in GENDER classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_gender, dev_conf_gender))
        print("Maximum confidences in AGE classification has mean %.4f and standard deviation %.4f" % (
            mean_conf_age, dev_conf_age))
        print("Entropy change has mean %.4f and standard deviation %.4f" % (
            mean_entropy_change, dev_entropy_change))

    # Params: action -> 0 or 1, states -> State objects
    def func(prev_state, action):
        c_g = prev_state.confidence('gender')
        c_a = prev_state.confidence('age')

        # TODO experiment with different cost weights
        reward_conf = .5 * (((c_g - mean_conf_gender) / dev_conf_gender) + 1) + \
                      .5 * (((c_a - mean_conf_age) / dev_conf_age) + 1)

        cost = init_cost + (prev_state.k_max - prev_state.k)
        if action == 1:
            reward = reward_conf - cost
        elif action == 0:
            # Punish with lost opportunity - or reward with it
            reward = 0 # reward_conf - 1

        return reward

    return func

""" some hardcoded data/constants + utility functions for RL env setup """

from functools import partial

import doy
import gym
import numpy as np
from procgen import ProcgenEnv


def normalize_return(ep_ret, env_name):
    """normalizes returns based on URP and expert returns above"""
    return doy.normalize_into_range(
        lower=urp_ep_return[env_name],
        upper=expert_ep_return[env_name],
        v=ep_ret,
    )


def setup_procgen_env(num_envs, env_id, gamma):
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
    )

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    envs.normalize_return = partial(normalize_return, env_name=env_id)
    return envs


ta_dim = {
    "bigfish": 15,
    "bossfight": 15,
    "caveflyer": 15,
    "chaser": 15,
    "climber": 15,
    "coinrun": 15,
    "dodgeball": 15,
    "fruitbot": 15,
    "heist": 15,
    "jumper": 15,
    "leaper": 15,
    "maze": 15,
    "miner": 15,
    "ninja": 15,
    "plunder": 15,
    "starpilot": 15,
}

procgen_names = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


procgen_action_meanings = np.array(
    [
        "LEFT-DOWN",
        "LEFT",
        "LEFT-UP",
        "DOWN",
        "NOOP",
        "UP",
        "RIGHT-DOWN",
        "RIGHT",
        "RIGHT-UP",
        "D",
        "A",
        "W",
        "S",
        "Q",
        "E",
    ]
)

# mean episodic returns for procgen (easy) under uniform random policy
urp_ep_return = {
    "bigfish": 0.8742888,
    "bossfight": 0.04618272,
    "caveflyer": 2.5970738,
    "chaser": 0.6632482,
    "climber": 2.2242901,
    "coinrun": 2.704834,
    "dodgeball": 0.612983,
    "fruitbot": -2.5330205,
    "heist": 3.0758767,
    "jumper": 3.4105318,
    "leaper": 2.5105245,
    "maze": 4.2726293,
    "miner": 1.2513667,
    "ninja": 2.5599792,
    "plunder": 4.3207445,
    "starpilot": 1.5251881,
}

# mean episodic returns for procgen (easy) under expert policy (from expert data)
expert_ep_return = {
    "bigfish": 36.336166,
    "bossfight": 11.634365,
    "caveflyer": 9.183605,
    "chaser": 9.955711,
    "climber": 10.233627,
    "coinrun": 9.93251,
    "dodgeball": 13.486584,
    "fruitbot": 29.925259,
    "heist": 9.685265,
    "jumper": 8.460201,
    "leaper": 7.4082565,
    "maze": 9.969294,
    "miner": 11.892558,
    "ninja": 9.474582,
    "plunder": 11.460528,
    "starpilot": 66.98625,
}

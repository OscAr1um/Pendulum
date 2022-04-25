# Author: Rui Liu

from __future__ import annotations

import numpy as np

from pendulum import PendulumEnv


env = PendulumEnv()

MAX_SPEED = env.max_speed
MAX_TORQUE = env.max_torque

def inspect(states: np.ndarray) -> np.ndarray:
    """
    Args:
        states (np.ndarray)
    Returns:
        indicator (np.ndarray): 1. if the states cause damage, and
                                0. if the states is safe
    """
    x, _, _ = states.T
    return np.where(x < 0, 1., 0.)
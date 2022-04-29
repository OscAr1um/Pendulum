# Author: Rui Liu

from __future__ import annotations

import numpy as np

from environment.pendulum import PendulumEnv


env = PendulumEnv()

MAX_SPEED = env.max_speed
MAX_TORQUE = env.max_torque

def inspect(state: np.ndarray) -> np.float32:
    """
    Args:
        state (np.ndarray)
    Returns:
        indicator (np.float32): 1. if the states cause damage, and
                                0. if the states is safe
    """
    x, _, _ = state
    return np.array(x < 0, dtype='float32')
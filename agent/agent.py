# Author: Rui Liu

from __future__ import annotations

from typing import Any

import numpy as np

from environment import MAX_TORQUE


class Agent(object):
    def __init__(self) -> None:
        pass
    
    def _soft_update(self, model: Any, target_model: Any, tau: float = .001) -> None:
        """
            θ_target = τ * θ + (1 - τ) * θ_target
        """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self) -> np.ndarray:
        """
        Args:
            state (np.ndarray)
        Return:
            action (np.ndarray)
        """
        return np.random.uniform(-MAX_TORQUE, MAX_TORQUE, 1).astype("float32")
    
    def step(self) -> None:
        pass

    def save_model(self, file_name: str):
        pass
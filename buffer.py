# Author: Rui Liu

from __future__ import annotations

from collections import deque
import random

import torch


class ExperienceReplay(object):
    """Replay Buffer (state, action, reward, next_state, done)."""
    def __init__(self, device: torch.device, buffer_size: int = 100000, batch_size: int = 128) -> None:
        """
        Args:
            device (torch.device): device to push batches
            buffer_size (int): maximum size of buffer
            batch_size (int): size of training batch
        """
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.attr_num = 0

    def add(self, *args: torch.Tensor) -> None:
        """
        Replay Buffer 
        Example:
            (state, action, reward, next_state, done)
        """
        if len(self.buffer) == 0:
            self.attr_num = len(args)
        assert len(args) == self.attr_num
        self.buffer.append(args)
    
    def sample(self) -> list[torch.Tensor]:
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(self.buffer, self.batch_size)
            transitions = [[] for _ in range(self.attr_num)]
            for transition in batch:
                for i, val in enumerate(transition):
                    transitions[i].append(val)
            return [torch.stack(attr).to(self.device) for attr in transitions]
        return []

    def __len__(self) -> int:
        return len(self.buffer)
#!/usr/bin/env python3
# Author: Rui Liu

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from environment import MAX_TORQUE
from buffer import ExperienceReplay
from agent import Agent
from train import train


class DQN(nn.Module):
    """DQN Model."""
    def __init__(self, num_actions) -> None:
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, state):
        return self.net(state)


class DQNAgent(Agent):
    """DQN Agent."""
    def __init__(self, device: torch.device, num_actions: int = 128, gamma: float = .99, epsilon: float = .1) -> None:
        self.device = device
        self.actions = np.linspace(-MAX_TORQUE, MAX_TORQUE, num_actions, dtype="float32")
        self.gamma = gamma
        self.epsilon = epsilon

        # DQN
        self.dqn = DQN(num_actions).to(self.device)
        self.dqn_target = DQN(num_actions).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=.001)

        # Replay Buffer
        self.buffer = ExperienceReplay(self.device)

    def act(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Args:
            state (np.ndarray)
            training (bool)
        Return:
            action (np.ndarray)
        """
        if training and random.choices([True, False], [self.epsilon, 1 - self.epsilon]):
            return np.random.choice(self.actions, 1)
        else:
            self.dqn.eval()
            with torch.no_grad():
                state = torch.from_numpy(state)
                qs = self.dqn(state.to(self.device)).cpu()
                index = torch.argmax(qs)
            return np.expand_dims(self.actions[index], -1)
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: bool) -> None:
        state, index, reward, next_state, done = (torch.from_numpy(state),
                                                  torch.from_numpy(np.argwhere(self.actions == action).ravel()),
                                                  torch.tensor(reward, dtype=torch.float),
                                                  torch.from_numpy(next_state),
                                                  torch.tensor(1.) if done else torch.tensor(0.))
        self.buffer.add(state, index, reward, next_state, done)

        samples = self.buffer.sample()
        if samples:
            states, indices, rewards, next_states, dones = samples
            self.dqn.train()

            # update dqn
            next_q = torch.max(self.dqn_target(next_states), dim=1).values
            q_target = rewards + (self.gamma * next_q * (1 - dones))
            q = torch.gather(self.dqn(states), 1, indices).squeeze()
            loss = F.mse_loss(q, q_target)
            logging.debug(f"Deep Q-Network is updating with loss={loss}.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target networks
            logging.debug(f"Updating target network.")
            self._soft_update(self.dqn, self.dqn_target)
    
    def save_model(self, file_name: str):
        torch.save(self.dqn.state_dict(), f"{file_name}.dqn")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device)
    train(agent, "DQN", num_episodes=200)
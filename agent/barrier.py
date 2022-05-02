#!/usr/bin/env python3
# Author: Rui Liu

from __future__ import annotations
from json import load

import logging
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from environment import MAX_TORQUE
from framework import ExperienceReplay, train_with_inspect
from agent.agent import Agent


class JudgeModule(nn.Module):
    """Judge Module."""
    def __init__(self, num_actions) -> None:
        super(JudgeModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.net(state)


class JudgeAgent(Agent):
    """Judge Agent."""
    def __init__(self, device: torch.device, num_actions: int = 64, gamma: float = .99, 
                 epsilon: float = .1, model_path: str = "") -> None:
        self.device = device
        self.actions = np.linspace(-MAX_TORQUE, MAX_TORQUE, num_actions, dtype="float32")
        self.gamma = gamma
        self.epsilon = epsilon

        # Judge Module
        self.judge = JudgeModule(num_actions).to(self.device)
        if model_path:
            self.judge.load_state_dict(torch.load(model_path))
        self.judge_target = JudgeModule(num_actions).to(self.device)
        self.judge_target.load_state_dict(self.judge.state_dict())
        self.optimizer = optim.Adam(self.judge.parameters(), lr=.001)

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
        self.judge.eval()
        with torch.no_grad():
            state = torch.from_numpy(state)
            js = self.judge(state.to(self.device)).cpu()
        if training and random.choices([True, False], [self.epsilon, 1 - self.epsilon]):
            indices = (js > .5).nonzero()
            if len(indices) > 0:
                index = random.choice(indices).squeeze()
                return np.expand_dims(self.actions[index], -1)
        index = torch.argmax(js)
        return np.expand_dims(self.actions[index], -1)
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, damage: np.ndarray, 
             next_state: np.ndarray, done: bool) -> None:
        state, index, damage, next_state, done = (torch.from_numpy(state),
                                                  torch.from_numpy(np.argwhere(self.actions == action).ravel()),
                                                  torch.tensor(damage, dtype=torch.float),
                                                  torch.from_numpy(next_state),
                                                  torch.tensor(1.) if done else torch.tensor(0.))
        self.buffer.add(state, index, damage, next_state, done)

        samples = self.buffer.sample()
        if samples:
            states, indices, damages, next_states, dones = samples
            self.judge.train()

            # update judge module
            next_j = torch.max(self.judge_target(next_states), dim=1).values
            j_target = (1 - damages) * next_j ** (1 - dones)
            j = torch.gather(self.judge(states), 1, indices).squeeze()
            loss = nn.BCELoss()
            loss = loss(j, j_target)
            logging.debug(f"Judge Module is updating with loss={loss}.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target networks
            logging.debug(f"Updating target network.")
            self._soft_update(self.judge, self.judge_target)
    
    def save_model(self, file_name: str):
        torch.save(self.judge.state_dict(), f"{file_name}.judge")


class TrigramJudgeAgent(JudgeAgent):
    """Trigram Judge Agent."""
    def __init__(self, device: torch.device, c: float = .9999, num_actions: int = 64, 
                 gamma: float = .99, epsilon: float = .1, model_path: str = "") -> None:
        super().__init__(device, num_actions, gamma, epsilon, model_path)
        self.c = c
        self.cache = deque(maxlen=3)
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, damage: np.ndarray, 
             next_state: np.ndarray, done: bool) -> None:
        state, index, damage, next_state, done = (torch.from_numpy(state),
                                                  torch.from_numpy(np.argwhere(self.actions == action).ravel()),
                                                  torch.tensor(damage, dtype=torch.float),
                                                  torch.from_numpy(next_state),
                                                  torch.tensor(1.) if done else torch.tensor(0.))
        self.cache.append((state, index, damage, next_state, done))
        if len(self.cache) == 3:
            self.buffer.add(self.cache[0][0], self.cache[0][1], self.cache[0][2], self.cache[1][0], self.cache[0][4], self.cache[2][0])

        samples = self.buffer.sample()
        if samples:
            states, indices, damages, next_states, dones, next_next_states = samples
            self.judge.train()

            # update judge module
            next_j = torch.max(self.judge_target(next_states), dim=1).values
            next_next_j = torch.max(self.judge_target(next_next_states), dim=1).values
            next_j = torch.where(next_next_j > self.c, torch.tensor(1.), next_j)
            j_target = (1 - damages) * next_j ** (1 - dones)
            j = torch.gather(self.judge(states), 1, indices).squeeze()
            loss = nn.BCELoss()
            loss = loss(j, j_target)
            logging.debug(f"Judge Module is updating with loss={loss}.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target networks
            logging.debug(f"Updating target network.")
            self._soft_update(self.judge, self.judge_target)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = TrigramJudgeAgent(device)
    train_with_inspect(agent, "TrigramJudge", num_episodes=20000)
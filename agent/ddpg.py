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
from framework import ExperienceReplay, train
from agent.agent import Agent


class Actor(nn.Module):
    """Policy Model."""
    def __init__(self) -> None:
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state) * MAX_TORQUE


class Critic(nn.Module):
    """Value Model."""
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(1 + 256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        state = F.relu(self.fcs1(state))
        q = torch.cat((state, action), dim=1)
        q = F.relu(self.fc2(q))
        return self.fc3(q)


class DDPGAgent(Agent):
    """DDPG Agent."""
    def __init__(self, device: torch.device, gamma: float = .99, epsilon: float = .1, actor_path: str = "", critic_path: str = "") -> None:
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Actor
        self.actor = Actor().to(self.device)
        if actor_path:
            self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=.001, weight_decay=.0001)

        # Critic
        self.critic = Critic().to(self.device)
        if critic_path:
            self.critic.load_state_dict(torch.load(critic_path))
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=.001, weight_decay=.0001)

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
            return np.random.uniform(-MAX_TORQUE, MAX_TORQUE, 1).astype('float32')
        else:
            self.actor.eval()
            with torch.no_grad():
                state = torch.from_numpy(state)
                action = self.actor(state.to(self.device)).cpu()
            return action.detach().numpy()
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: bool) -> None:
        state, action, reward, next_state, done = (torch.from_numpy(state),
                                                   torch.from_numpy(action),
                                                   torch.tensor(reward, dtype=torch.float).unsqueeze(-1),
                                                   torch.from_numpy(next_state),
                                                   torch.tensor([1.]) if done else torch.tensor([0.]))
        self.buffer.add(state, action, reward, next_state, done)

        samples = self.buffer.sample()
        if samples:
            states, actions, rewards, next_states, dones = samples
            self.actor.train()
            
            # update critic
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            q_target = rewards + (self.gamma * next_q * (1 - dones))
            q = self.critic(states, actions)
            critic_loss = F.mse_loss(q, q_target)
            logging.debug(f"Critic Network is updating with loss={critic_loss}.")
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # update actor
            actions = self.actor(states)
            actor_loss = -self.critic(states, actions).mean()
            logging.debug(f"Actor Network is updating with loss={actor_loss}.")
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            logging.debug(f"Updating target network.")
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
    
    def save_model(self, file_name: str):
        torch.save(self.actor.state_dict(), f"{file_name}.actor")
        torch.save(self.critic.state_dict(), f"{file_name}.critic")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(device)
    train(agent, "DDPG", num_episodes=200)
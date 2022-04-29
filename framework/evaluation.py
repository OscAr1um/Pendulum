# Author: Rui Liu

from __future__ import annotations

from typing import Union, Any
from math import nan
from collections import OrderedDict

import numpy as np
import torch

from environment import env, MAX_SPEED, inspect


class Score(object):
    def __init__(self) -> None:
        self.scores: OrderedDict[str, list] = OrderedDict()
        self.fields = ['Episode', 'Time', 'Iteration', 'Reward']
        for field in self.fields:
            self.scores[field] = []
        self._size = 0

    def append(self, episode: int, time: int, iteration: float, reward: float):
        self.scores['Episode'].append(episode)
        self.scores['Time'].append(time)
        self.scores['Iteration'].append(iteration)
        self.scores['Reward'].append(reward)
        self._size += 1
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index: Union[int, slice]) -> OrderedDict:
        items = OrderedDict()
        for field in self.scores:
            items[field] = self.scores[field][index]
        return items
    
    def save_scores(self, file_name: str) -> None:
        with open(file_name, "w") as f:
            for i in range(self._size):
                line = ""
                for field in self.scores:
                    line += f"{self.scores[field][i]}\t"
                f.write(line[:-1] + "\n")
        f.close()


class ScoreWithInspect(Score):
    def __init__(self, judge = None) -> None:
        super().__init__()
        self._states = self._generate_default_states()
        _extended_fields = ["Average Iteration", "Average Reward", 
                            "Number of Safe Inits", "Average Iteration from Safe Inits", "Average Reward from Safe Inits", 
                            "Number of Unsafe Inits", "Average Iteration from Unsafe Inits", "Average Reward from Unsafe Inits"]
        for field in _extended_fields:
            self.scores[field] = []
        self.fields.extend(_extended_fields)
        self.judge = judge if judge else lambda _: 1
    
    def _generate_default_states(self) -> np.ndarray:
        Theta = np.linspace(-np.pi, np.pi, 25)
        Omega = np.linspace(-MAX_SPEED, MAX_SPEED, 33)
        Theta, Omega = np.meshgrid(Theta[:25], Omega)
        return np.stack([Theta.flatten(), Omega.flatten()]).T
    
    def append(self, episode: int, time: int, iteration: float, reward: float, agent: Any, replace_judge: bool):
        super().append(episode, time, iteration, reward)
        self._calculate_score(agent, replace_judge)
    
    def _calculate_score(self, agent, replace_judge: bool = False) -> None:
        if replace_judge:
            self.judge = agent.judge
            agent.judge.eval()
        safe_iteration = []
        safe_reward = []
        unsafe_iteration = []
        unsafe_reward = []
        for init_state in self._states:
            env.reset()
            env.state = init_state
            state = env.get_obs()
            safe = True if self.judge(torch.from_numpy(state)).max() > .5 else False
            i = 0
            r = 0
            if not inspect(state):
                for i in range(200):
                    action = agent.act(state, training=False)
                    next_state, reward, done, _ = env.step(action)
                    damage = inspect(next_state)
                    r += reward
                    if damage or done:
                        break
                    state = next_state
            if safe:
                safe_iteration.append(i)
                safe_reward.append(r)
            else:
                unsafe_iteration.append(i)
                unsafe_reward.append(r)
        self.scores["Average Iteration"].append((sum(safe_iteration) + sum(unsafe_iteration)) / (len(safe_iteration) + len(unsafe_iteration)))
        self.scores["Average Reward"].append((sum(safe_reward) + sum(unsafe_reward)) / (len(safe_reward) + len(unsafe_reward)))
        self.scores["Number of Safe Inits"].append(len(safe_iteration))
        self.scores["Average Iteration from Safe Inits"].append(sum(safe_iteration) / len(safe_iteration) if safe_iteration else nan)
        self.scores["Average Reward from Safe Inits"].append(sum(safe_reward) / len(safe_reward) if safe_reward else nan)
        self.scores["Number of Unsafe Inits"].append(len(unsafe_iteration))
        self.scores["Average Iteration from Unsafe Inits"].append(sum(unsafe_iteration) / len(unsafe_iteration) if unsafe_iteration else nan)
        self.scores["Average Reward from Unsafe Inits"].append(sum(unsafe_reward) / len(unsafe_reward) if unsafe_reward else nan)
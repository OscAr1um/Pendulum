# Author: Rui Liu

from __future__ import annotations

from typing import Any
import os
import logging
import time
from collections import namedtuple

from environment import env


Score = namedtuple('Score', ['Episode', 'Time', 'Iteration', 'Reward'])


def save_scores(scores: list, file_name: str) -> None:
    with open(file_name, "w") as f:
        for episode, *score in scores:
            line = f"{episode}"
            for value in score:
                line += f"\t{value}"
            line += "\n"
            f.write(line)
    f.close()


def train(agent: Any, model_name: str, num_episodes: int = 10000) -> list[dict]:
    scores = []
    if not os.path.isdir(model_name):
        os.mkdir(model_name)
    start = time.time()
    for episode in range(num_episodes):
        state = env.reset()
        r = 0
        for i in range(200):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            r += reward
            if done:
                break
            state = next_state
        score = Score(Episode=episode, Time=int(time.time()-start), Iteration=i, Reward=r)
        scores.append(score)
        if (episode + 1) % 10 == 0:
            agent.save_model(f"{model_name}/{episode}")
            logging.info(score)
    agent.save_model(f"{model_name}")
    save_scores(scores, f"{model_name}.csv")
    return scores
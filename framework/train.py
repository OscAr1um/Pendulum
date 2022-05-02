# Author: Rui Liu

from __future__ import annotations

from typing import Any
import os
import logging
import time

from environment import env, inspect
from framework.evaluation import Score, ScoreWithInspect


if not os.path.isdir("resource"):
    os.mkdir("resource")
if not os.path.isdir("model"):
    os.mkdir("model")
if not os.path.isdir("score"):
    os.mkdir("score")


def train(agent: Any, model_name: str, num_episodes: int = 10000) -> Score:
    scores = Score()
    if not os.path.isdir(f"resource/{model_name}"):
        os.mkdir(f"resource/{model_name}")
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
        scores.append(episode=episode, time=int(time.time()-start), iteration=i, reward=r)
        if (episode + 1) % 10 == 0:
            agent.save_model(f"resource/{model_name}/{episode}")
            logging.info(scores[-1])
    agent.save_model(f"model/{model_name}")
    scores.save_scores(f"score/{model_name}.csv")
    print(scores[-1])
    return scores


def train_with_inspect(agent: Any, model_name: str, num_episodes: int = 10000, judge: Any = None) -> ScoreWithInspect:
    scores = ScoreWithInspect(judge=judge)
    replace_judge = False if judge else True
    if not os.path.isdir(f"resource/{model_name}"):
        os.mkdir(f"resource/{model_name}")
    start = time.time()
    for episode in range(num_episodes):
        state = env.reset()
        r = 0
        i = 0
        if not inspect(state):
            for i in range(200):
                action = agent.act(state, training=True)
                next_state, reward, done, _ = env.step(action)
                damage = inspect(next_state)
                agent.step(state, action, reward, damage, next_state, done)
                r += reward
                if damage or done:
                    break
                state = next_state
        if (episode + 1) % 10 == 0:
            agent.save_model(f"resource/{model_name}/{episode}")
            scores.append(episode=episode, time=int(time.time()-start), iteration=i, reward=r, agent=agent, replace_judge=replace_judge)
            logging.info(scores[-1])
    agent.save_model(f"model/{model_name}")
    scores.save_scores(f"score/{model_name}.csv")
    print(scores[-1])
    return scores
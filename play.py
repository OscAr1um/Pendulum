# Author: Rui Liu

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from environment import env, inspect
from agent import DDPGAgent, DQNAgent, AssuredDQNAgent, JudgeAgent, TrigramJudgeAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pendulum")
    parser.add_argument(
        "-i",
        "--inspection",
        type=bool,
        help="Inspection",
        default=False,
    )
    parser.add_argument(
        "-a",
        "--agent", 
        type=str, required=True, 
        help="Agent",
    )
    parser.add_argument(
        "--actor-model", 
        type=str,
        help="Path of actor model",
        default="model/DDPG.actor",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        help="Path of critic model",
        default="model/DDPG.critic",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path of model",
        default="model/DQN.dqn",
    )
    parser.add_argument(
        "-j",
        "--judge-agent", 
        type=str, 
        help="Judge agent",
        default=""
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Path of judge model",
        default="model/TrigramJudge.judge",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode",
        default="human",
    )
    parser.add_argument(
        "--init-state",
        type=str,
        help="Initial state",
        default="[0.,0.]",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration",
        default=0,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Output path",
        default="image/default.obj",
    )
    return parser.parse_args()


def run(agent, mode: str = "human", inspection: bool = False, initial_state: str = "[0.,0.]", 
        iteration: int = 0, output_path: str = "image/default.obj") -> None:
    assert mode in {"human", "rgb_array"}
    env.reset()
    if mode == "rgb_array":
        env.state = np.array(eval(initial_state))
    state = env.get_obs()
    for i in range(200):
        action = agent.act(state)
        if mode == "human":
            env.render()
            time.sleep(.5)
        elif i == iteration:
            env.render(mode="rgb_array", image_path=output_path)
            break
        next_state, _, done, _ = env.step(action)
        d = False if inspection else inspect(next_state)
        if d or done:
            break
        state = next_state
    env.close()
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    if args.agent == "DDPG":
        agent = DDPGAgent(device, actor_path=args.actor_model, critic_path=args.critic_model)
    elif args.agent == "DQN":
        agent = DQNAgent(device, model_path=args.model)
    elif args.agent == "AssuredDQN":
        if args.judge_agent == "Judge":
            judge = JudgeAgent(device, model_path=args.judge_model).judge
        elif args.judge_agent == "TrigramJudge":
            judge = TrigramJudgeAgent(device, model_path=args.judge_model).judge
        else:
            judge = None
        agent = AssuredDQNAgent(device, judge=judge, model_path=args.model)
    run(agent, args.mode, args.inspection, args.init_state, args.iteration, args.output_path)
# Pendulum
An Implementation of Reinforcement Learning

## Environment
- Modified the name of the private attribute `_get_obs()` in the original environment to enable evaluations.
- As an optional constrain, the inspector returns $0$ when the pendulum is below the horizen, meaning a damage $d=1$ is detected, while $1$ is returned otherwise.

## Algorithm
- DDPG
- DQL
- Barrier Learner
- Constraint RL
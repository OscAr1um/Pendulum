# Pendulum
An Implementation of Reinforcement Learning

## Environment
- Changed the name of the private attribute `_get_obs()` to `get_obs()` in the original environment to enable evaluations.
- Modified `render()` function to output image RGB array.
- As an optional constrain, `inspect()` returns 0 when the pendulum is below the horizen, meaning a damage d=1 is detected, while 1 is returned otherwise.
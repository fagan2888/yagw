# Yet Another Grid World

  [**Installation**](#installation)
| [**Get Started**](#get-started)
| [**Dependencies**](#dependencies)

YAGW (pronounced "yoo") is a simple grid world (a.k.a. maze) environment, implementing the [`dm_env.Environment`](https://github.com/deepmind/dm_env) interface.

If you find this open source release useful, please cite in your paper:

```
@misc{yagw,
  author = {Filos, Angelos},
  title = {Yet Another Grid World},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/filangel/yagw}},
}
```

## Installation

You can install YAGW with `pip` by running:

```shell
$ pip install git+https://github.com/filangel/yagw.git
```

## Get Started

You can instantiate a simple environment by running:

```python
from yagw.dm_env_interface import GridWorld

# Init a simple, single maze environment.
env = GridWorld(num_layouts=1, max_steps_count=50)
timestep = env.reset()

# RL loop:
while not timestep.last():
  # Random action.
  action = env.action_spec().generate_value()
  # Step in the simulator.
  timestep = env.step(action)
  # Render video.
  env.render(mode="human")
```

## Dependencies

The code was tested under Ubuntu 18.04 and macOS with the Miniconda Python distribution and uses these packages (as listed in [`setup.py`](./setup.py)):

* [`labmaze==1.0.2`](https://pypi.org/project/labmaze/1.0.2/)
* [`dm-env==1.2`](https://pypi.org/project/dm-env/1.2/)
* [`pycolab==1.2`](https://pypi.org/project/pycolab/1.2/)
* [`matplotlib==3.2.1`](https://pypi.org/project/matplotlib/3.2.1/)
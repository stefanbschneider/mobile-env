[![CI](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-package.yml/badge.svg)](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-package.yml)
[![PyPI](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-publish.yml/badge.svg)](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-publish.yml)
[![Documentation](https://readthedocs.org/projects/mobile-env/badge/?version=latest)](https://mobile-env.readthedocs.io/en/latest/?badge=latest)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/demo.ipynb)


# mobile-env: An Open Environment for Autonomous Coordination in Mobile Networks

mobile-env is an open, minimalist OpenAI Gym environment for training and evaluating coordination algorithms in wireless mobile networks. 
The environment allows modeling users moving around an area and can connect to one or multiple base stations.
Using the Gym interface, the environment can be used with any reinforcement learning framework (e.g., stable-baselines or Ray RLlib) or any custom (even non-RL) coordination approach.
The environment is highly configurable and can be easily extended (e.g., regarding users, movement patterns, channel models, etc.).

mobile-env supports multi-agent and centralized reinforcement learning policies. It provides various choices for rewards and observations. mobile-env is also easily extendable, so that anyone may add another channel models (e.g. path loss), movement patterns, utility functions, etc.

As an example, mobile-env can be used to study multi-cell selection in coordinated multipoint.
Here, it must be decided what connections should be established among user equipments (UEs) and base stations (BSs) in order to maximize Quality of Experience (QoE) globally. 
To maximize the QoE of single UEs, the UE intends to connect to as many BSs as possible, which yields higher (macro) data rates. 
However, BSs multiplex resources among connected UEs (e.g. schedule physical resource blocks) and, therefore, UEs compete for limited resources (conflicting goals). 
To maximize QoE globally, the policy must recognize that (1) the data rate of any connection is governed by the channel (e.g. SNR) between UE and BS and (2) QoE of single UEs not necessarily grows linearly with increasing data rate.

<p align="center">
    <img src="https://user-images.githubusercontent.com/36734964/139288123-7732eff2-24d4-4c25-87fd-ac906f261c93.gif" width="65%"/>
    <br>
    <sup><a href="https://thenounproject.com/search/?q=base+station&i=1286474" target="_blank">Base station icon</a> by Clea Doltz from the Noun Project</sup>
</p>

**[Try mobile-env: ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/demo.ipynb)**

Documentation and API: [ReadTheDocs](https://mobile-env.readthedocs.io/en/latest/)


## Installation

### From PyPI (Recommended)

The simplest option is to install the latest release of `mobile-env` from [PyPI](https://pypi.org/project/mobile-env/) using pip:

```bash
pip install mobile-env
```
This is recommended for most users. mobile-env is tested on Ubuntu, Windows, and MacOS.

### From Source (Development)

Alternatively, for development, you can clone `mobile-env` from GitHub and install it from source.
After cloning, install in "editable" mode (-e):

```bash
pip install -e .
```

This is equivalent to running `pip install -r requirements.txt`.

If you want to run tests or examples, also install the requirements in `requirements_tests.txt`.
For dependencies for building docs, install the requirements in `requirements_docs.txt`.

## Example Usage

```python
import gym
import mobile_env

env = gym.make("mobile-medium-central-v0")
obs = env.reset()
done = False

while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```

## Customization

mobile-env supports custom channel models, movement patterns, arrival & departure models, resource multiplexing schemes and utility functions. 
For example, replacing the default [Okumura–Hata](https://en.wikipedia.org/wiki/Hata_model) channel model by a (simplified) path loss model can be as easy as this:

```python
import gym
import numpy as np
from mobile_env.core.base import MComCore
from mobile_env.core.channel import Channel


class PathLoss(Channel):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        # path loss exponent
        self.gamma = gamma

    def power_loss(self, bs, ue):
        """Computes power loss between BS and UE."""
        dist = bs.point.distance(ue.point)
        loss = 10 * self.gamma * np.log10(4 * np.pi * dist * bs.frequency)
        return loss


# replace default channel model in configuration 
config = MComCore.default_config()
config['channel'] = PathLoss

# pass init parameters to custom channel class!
config['channel_params'].update({'gamma': 2.0})

# create environment with custom channel model
env = gym.make('mobile-small-central-v0', config=config)
# ...
```


## About `mobile-env`


If you use mobile-env in your work, please cite with:

```
@misc{mobile-env,
  author = {Schneider, Stefan and Werner, Stefan and Karl, Holger},
  title = {An Open Environment for Autonomous Coordination in Mobile Networks},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/stefanbschneider/mobile-env}},
}
```

mobile-env is based on [DeepCoMP](https://github.com/CN-UPB/DeepCoMP), providing the underlying environment as open, stand-alone platform to be used by others.


List of repositories, publications, or preprints using `mobile-env` (please open a pull request to add missing entries):

* [DeepCoMP](https://github.com/CN-UPB/DeepCoMP)



## Contributing

Development: [@stefanbschneider](https://github.com/stefanbschneider) and [@stwerner97](https://github.com/stwerner97/)


We happy if you find `mobile-env` useful. If you have feedback or want to report bugs, feel free to [open an issue](https://github.com/stefanbschneider/mobile-env/issues/new).

We also welcome contributions: Whether you implement a new channel model, fix a bug, or just make a minor addition elsewhere, feel free to open a pull request!



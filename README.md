[![Python package](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-package.yml/badge.svg)](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/mobile-env/badge/?version=latest)](https://mobile-env.readthedocs.io/en/latest/?badge=latest)
[![Publish](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-publish.yml/badge.svg)](https://github.com/stefanbschneider/mobile-env/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/mobile-env)](https://pypi.org/project/mobile-env/)

#  [Try Mobile-Env on Google Colab!   ![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/tutorial.ipynb)
Mobile-Env is a minimalist OpenAI-Gym environment for training and evaluating intelligent coordination algorithms in wireless mobile networks. At each time step, it must be decided what connections should be established among user equipments (UEs) and basestations (BSs) in order to maximize Quality of Experience (QoE) globally. To maximize the QoE of single UEs, the UE intends to connect to as many BSs as possible, which yields higher (macro) data rates. However, BSs multiplex resources among connected UEs (e.g. schedule physical resource blocks) and, therefore, UEs compete for limited resources (conflicting goals). To maximize QoE globally, the policy must recognize that (1) the data rate of any connection is governed by the channel (e.g. SNR) between UE and BS and (2) QoE of single UEs not necessarily grows linearly with increasing data rate.

Mobile-Env supports multi-agent and centralized reinforcement learning policies. It provides various choices for rewards and observations. Mobile-Env is also easily extendable, so that anyone may add another channel models (e.g. path loss), movement patterns, utility functions, etc.

<p align="center">
    <img src="https://user-images.githubusercontent.com/36734964/139288123-7732eff2-24d4-4c25-87fd-ac906f261c93.gif" width="65%"/>
</p>


## Installation
```bash
pip install mobile-env
```

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

## Customizability
Mobile-Env supports custom channel models, movement patterns, arrival & departure models, resource multiplexing schemes and utility functions. For example, replacing the default [Okumura–Hata](https://en.wikipedia.org/wiki/Hata_model) channel model by a (simplified) path loss model can be as easy as this:
```python
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
...
```

## Documentation
Read the [documentation online](https://mobile-env.readthedocs.io/en/latest/index.html).


## DeepCoMP
Mobile-Env builds on [DeepCoMP](https://github.com/CN-UPB/DeepCoMP) and makes its simulation accessible via an OpenAI Gym interface that is independent of any specific reinforcement learning framework.

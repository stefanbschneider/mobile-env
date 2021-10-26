![GitHub Workflow Status](https://img.shields.io/github/workflow/status/stefanbschneider/mobile-env/Python%20package)
[![Documentation Status](https://readthedocs.org/projects/mobile-env/badge/?version=latest)](https://mobile-env.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/mobile-env)](https://pypi.org/project/mobile-env/)
![GitHub all releases](https://img.shields.io/github/downloads/stefanbschneider/mobile-env/total)

# Mobile-Env
Mobile-Env is a minimalist OpenAI-Gym environment for training and evaluating intelligent coordination algorithms in wireless mobile networks. At each time step, it must be decided what connections should be established among user equipments (UEs) and basestations (BSs) in order to maximize Quality of Experience (QoE) globally. To maximize the QoE of single UEs, the UE intends to connect to as many BSs as possible, which yields higher (macro) data rates. However, BSs multiplex resources among connected UEs (e.g. schedule physical resource blocks) and, therefore, UEs compete for limited resources (conflicting goals). To maximize QoE globally, the policy must recognize that (1) the data rate of any connection is governed by the channel (e.g. SNR) between UE and BS and (2) QoE of single UEs not necessarily grows linearly with increasing data rate.

Mobile-Env supports multi-agent and centralized reinforcement learning policies. It provides various choices for rewards and observations. Mobile-Env is also easily extendable, so that anyone may add another channel models (e.g. path loss), movement patterns, utility functions, etc.

## [Try Mobile-Env on Google Colab!   ![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/tutorial.ipynb)

<!-- TODO: GIF -->

## Installation
`pip install mobile-env`

## Usage

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

## Documentation

Read the [documentation online](https://mobile-env.readthedocs.io/en/latest/index.html).
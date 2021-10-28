# Wrappers
Wrapper classes are used to make interfaces compatible with other libraries if they aren't already. For example, the multi-agent reinforcement learning libraries [RLlib](https://docs.ray.io/en/stable/rllib.html) and [PettingZoo](https://www.pettingzoo.ml/) expect different interfaces. So far, we only provide a wrapper class for RLlib's MultiAgentEnv, so that RLlib's multi-agent algorithms can be used to train on mobile-env!. 

Example usage of our **RLlibMAWrapper** class:
```python
from mobile_env.wrappers.multi_agent import RLlibMAWrapper

env = gym.make('mobile-small-ma-v0')
# wrap multi-agent env for RLlib compatibility
env = RLlibMAWrapper(env)
# use RLlib to train on the environment ...
...
```
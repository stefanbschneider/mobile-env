(examples)=

# Examples

## Examples on Google Colab
We provide an in-depth example of mobile-env's usage on Google Colab! The notebook shows how to train multi-agent reinforcement learning policies with [RLlib](https://docs.ray.io/en/stable/rllib.html) and centralized control policies with [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).


[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/tutorial.ipynb)

## Environment Creation
Here is an example of how mobile-env's environments can be created:
```python
import gym
import mobile_env

# small environment; centralized control 
env = gym.make('mobile-small-central-v0')

# large environment; centralized control 
env = gym.make('mobile-large-central-v0')

# small environment; multi-agent control 
env = gym.make('mobile-large-ma-v0')
...
```

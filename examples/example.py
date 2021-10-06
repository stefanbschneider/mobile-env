import gym
# import so that env registers itself
import mobile_env
import random
import time

env = gym.make("mobile-small-v0")


for _ in range(100):
    actions = [random.randint(0, 1) for _ in range(5)]
    env.step(actions)
    env.render()
    time.sleep(0.1)


env.close()

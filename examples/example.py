import gym
# import so that env registers itself
import mobile_env

env = gym.make("mobile-v0")
env.config["max_time"] = 3
env.reset()
for _ in range(5):
    action = 1
    obs, reward, done, info = env.step(action)
    env.render()

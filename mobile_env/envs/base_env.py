import gym


class BaseEnv(gym.Env):
    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            self.config.update(config)

        self.reset()

    @classmethod
    def default_config(cls):
        return {
            'max_time': 10,
        }

    def reset(self):
        self.time = 0
        self.max_time = self.config['max_time']

    def obs(self):
        return {'time': self.time}

    def reward(self):
        return 0

    def done(self):
        return None

    def info(self):
        return {'time': self.time}

    def step(self, action):
        self.time += 1
        obs = self.obs()
        reward = self.reward()
        done = self.done()
        info = self.info()
        return obs, reward, done, info

    def render(self, mode='human'):
        print(self.time)


gym.envs.register(
    id = "mobile-v0",
    entry_point = "mobile_env.envs:BaseEnv"
)

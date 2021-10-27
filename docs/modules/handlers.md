# Handlers
Handler classes implement the interface methods and properties for OpenAI Gym. While the [core simulation](https://mobile-env.readthedocs.io/en/latest/modules/core.html) implements the base cell assignment functionality, the handler class implements how agents may interact with it. In other words, it defines whether environment provides a multi-agent or centralized interface, what actions are expected and what observations and rewards are returned as feedback. To do so, the handler must at least implement the following (class)methods: `observation_space`, `action_space`, `action`, `observation` and `reward`. The handler class is passed to the simulation within the configuration. The core simulation calls its methods according to the *strategy pattern*.


An example for passing a custom handler to the environment:
```python
class CustomHandler(Handler):
    @classmethod
    def action_space(cls, env):
        ...

    @classmethod
    def observation_space(cls, env):
        ...

    @classmethod
    def action(cls, env, action):
        ...

    @classmethod
    def observation(cls, env):
        ...

    @classmethod
    def reward(cls, env):
        ...

config = {'handler': CustomHandler}
env = gym.make('mobile-small-central-v0', config=config)
```

So far, [mobile-env](https://mobile-env.readthedocs.io/en/latest/index.html) implements handlers for a multi-agent and centralized control setting.
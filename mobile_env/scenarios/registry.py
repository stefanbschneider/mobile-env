import itertools

import gym

from mobile_env.scenarios.small import MComSmall
from mobile_env.scenarios.medium import MComMedium
from mobile_env.scenarios.large import MComLarge
from mobile_env.handlers.multi_agent import MComMAHandler
from mobile_env.handlers.central import MComCentralHandler


scenarios = {"small": MComSmall, "medium": MComMedium, "large": MComLarge}
handlers = {"ma": MComMAHandler, "central": MComCentralHandler}

for scenario, handler in itertools.product(scenarios, handlers):
    env_name = scenarios[scenario].__name__
    config = {"handler": handlers[handler]}
    gym.envs.register(
        id=f"mobile-{scenario}-{handler}-v0",
        entry_point=f"mobile_env.scenarios.{scenario}:{env_name}",
        kwargs={"config": config},
    )

import itertools

import gymnasium

from mobile_env.handlers.smart_city_handler import MComSmartCityHandler
from mobile_env.scenarios.smart_city import MComSmartCity


scenarios = {"smart_city": MComSmartCity}
handlers = {"smart_city_handler": MComSmartCityHandler}

for scenario, handler in itertools.product(scenarios, handlers):
    env_name = scenarios[scenario].__name__
    config = {"handler": handlers[handler]}
    gymnasium.envs.register(
        id=f"mobile-{scenario}-{handler}-v0",
        entry_point=f"mobile_env.scenarios.{scenario}:{env_name}",
        kwargs={"config": config},
    )

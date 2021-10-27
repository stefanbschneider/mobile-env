# Core Simulation

The [base environment](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.base.MComCore) class simulates cell assignment of many user equipments (UEs) to possibly multiple basestations (BSs) in a mobile communication setting. Our simulation is mainly composed of five components that implement their functionality in a replacable manner:

1. [Channel](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.channels.Channel): computes the data rate of connections among UEs and BSs.
2. [Arrival](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.arrival.Arrival): specifies the arrival and departure times of UEs, i.e., defines when and how long UEs request service.
3. [Movement](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.movement.Movement): model to define the movement patterns of UEs. 
4. [Schedule](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.schedules.Scheduler): defines how BSs multiplex resources among all connected UEs.
5. [Utility Function](https://mobile-env.readthedocs.io/en/latest/source/mobile_env.core.html#mobile_env.core.utilities.Utility): defines a function that quantifies the quality of experience (QoE) of UEs dependend on their (macro) data rate.

All components are called according to the *strategy pattern*, i.e., they define a fixed set of methods invoked by the base environment. This improves the configurability and extendability of our project. For example, if we would like to change the simulation's default movement pattern so that a selectable UE doesn't move at all, it can be implemented as follows:
```python
from mobile_env.core.movement import RandomWaypointMovement

class LazyUEMovement(RandomWaypointMovement):
    def __init__(self, lazy_ue, **kwargs):
        super().__init__(**kwargs)
        # this UE doen't move!
        self.lazy_ue = lazy_ue
        
    def move(self, ue):
        """Overrides default movement pattern."""
        # do not move lazy UE!
        if ue.ue_id == self.lazy_ue:
            return ue.x, ue.y
        
        # default movement otherwise
        return super().move(ue)


from mobile_env.core.base import MComCore

# replace default movement in configuration 
config = MComCore.default_config()
config['movement'] = LazyUEMovement
# pass init parameters to custom movement class!
config['movement_params'].update({'lazy_ue': 3})

# create environment with lazy UE!
env = gym.make('mobile-small-central-v0', config=config)
```

This example demonstrates that each core component can be replaced **without changing source code**. It also shows that we can pass parameters to the components' initialization via the ``config`` dictionary. Adapting the channel model, ... works similarly! 
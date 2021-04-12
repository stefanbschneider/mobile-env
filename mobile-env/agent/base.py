"""Abstract base classes for central vs distributed/multi agents"""


class CentralAgent:
    """Single central agent that observes and controls all UEs at once."""
    def __init__(self):
        self.central_agent = True

    def compute_action(self, observation):
        raise NotImplementedError("This needs to be implemented in the child class")


class MultiAgent:
    def __init__(self):
        self.central_agent = False

    def compute_action(self, observation, policy_id):
        raise NotImplementedError("This needs to be implemented in the child class")

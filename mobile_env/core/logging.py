from typing import Dict, List

class Monitor:
    def __init__(self, scalar_metrics: Dict, ue_metrics: Dict, bs_metrics: Dict, **kwargs):
        
        self.scalar_metrics: Dict = scalar_metrics
        self.ue_metrics: Dict = ue_metrics
        self.bs_metrics: Dict = bs_metrics

        self.scalar_results: Dict = None
        self.ue_results: Dict = None
        self.bs_results: Dict = None

    def reset(self):
        """Reset tracked results for all metrics."""

        self.scalar_results = {name: [] for name in self.scalar_metrics}
        self.ue_results = {name: [] for name in self.ue_metrics}
        self.bs_results = {name: [] for name in self.bs_metrics}

    def update(self, simulation):
        """Evaluate metrics given the simulation state and update tracked results."""

        # evaluate scalar, ue, bs metrics by passing the simulation state
        scalar_updates = {name: metric(simulation) for name, metric in self.scalar_metrics.items()}
        ue_updates = {name: metric(simulation) for name, metric in self.ue_metrics.items()}
        bs_updates = {name: metric(simulation) for name, metric in self.bs_metrics.items()}

        # update results by appending the metrics' return values (scalars or lists)
        self.scalar_results = {name: self.scalar_results[name] + [scalar_updates[name]] for name in self.scalar_metrics}
        self.ue_results = {name: self.ue_results[name] + ue_updates[name] for name in self.ue_metrics}
        self.bs_results = {name: self.bs_results[name] + bs_updates[name] for name in self.bs_metrics}

    def info(self):
        """Outputs the latest results as a dictionary."""
        scalar_info = {name: values[-1] for name, values in self.scalar_results.items()}
        ue_info = {name: values[-1] for name, values in self.ue_results.items()}
        bs_info = {name: values[-1] for name, values in self.bs_results.items()}

        return {**scalar_info, **ue_info, **bs_info}

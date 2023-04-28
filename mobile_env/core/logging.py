from typing import Dict

import pandas as pd


class Monitor:
    def __init__(
        self, scalar_metrics: Dict, ue_metrics: Dict, bs_metrics: Dict, **kwargs
    ):

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
        """Evaluate and update metrics given the simulation state."""

        # evaluate scalar, ue, bs metrics by passing the simulation state
        scalar_updates = {
            name: metric(simulation) for name, metric in self.scalar_metrics.items()
        }
        ue_updates = {
            name: metric(simulation) for name, metric in self.ue_metrics.items()
        }
        bs_updates = {
            name: metric(simulation) for name, metric in self.bs_metrics.items()
        }

        # update results by appending the metrics' return values
        self.scalar_results = {
            name: self.scalar_results[name] + [scalar_updates[name]]
            for name in self.scalar_metrics
        }
        self.ue_results = {
            name: self.ue_results[name] + [ue_updates[name]] for name in self.ue_metrics
        }
        self.bs_results = {
            name: self.bs_results[name] + [bs_updates[name]] for name in self.bs_metrics
        }

    def load_results(self):
        """Outputs results of tracked metrics as data frames."""
        # load scalar results with index (metric; time)
        scalar_results = pd.DataFrame(self.scalar_results)
        scalar_results.index.names = ["Time Step"]

        # load UE results with index (metric, UE ID; time)
        ue_results = {
            (metric, ue_id): [values.get(ue_id) for values in entries]
            for metric, entries in self.ue_results.items()
            for ue_id in set().union(*entries)
        }
        ue_results = pd.DataFrame(ue_results).transpose()
        ue_results.index.names = ["Metric", "UE ID"]
        # change data frame format to align time axis along rows
        ue_results = ue_results.stack()
        ue_results.index.names = ["Metric", "UE ID", "Time Step"]
        ue_results = ue_results.reorder_levels(["Time Step", "UE ID", "Metric"])
        ue_results = ue_results.unstack()

        # load BS results with index (metric, BS ID; time)
        bs_results = {
            (metric, bs_id): [values.get(bs_id) for values in entries]
            for metric, entries in self.bs_results.items()
            for bs_id in set().union(*entries)
        }
        bs_results = pd.DataFrame(bs_results).transpose()
        bs_results.index.names = ["Metric", "BS ID"]
        # change data frame format to align time axis along rows
        bs_results = bs_results.stack()
        bs_results.index.names = ["Metric", "BS ID", "Time Step"]
        bs_results = bs_results.reorder_levels(["Time Step", "BS ID", "Metric"])
        bs_results = bs_results.unstack()

        return scalar_results, ue_results, bs_results

    def info(self):
        """Outputs the latest results as a dictionary."""

        # Return empty infos if there are no scalar results.
        if any(len(results) == 0 for results in self.scalar_results.values()):
            return {}

        scalar_info = {name: values[-1] for name, values in self.scalar_results.items()}
        ue_info = {name: values[-1] for name, values in self.ue_results.items()}
        bs_info = {name: values[-1] for name, values in self.bs_results.items()}

        return {**scalar_info, **ue_info, **bs_info}

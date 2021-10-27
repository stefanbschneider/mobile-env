import string
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict

import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pygame import Surface
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mobile_env.handlers.central import MComCentralHandler
from mobile_env.core.util import BS_SYMBOL
from mobile_env.core.arrival import NoDeparture
from mobile_env.core.channels import OkumuraHata
from mobile_env.core.schedules import ResourceFair
from mobile_env.core.utilities import BoundedLogUtility
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.movement import RandomWaypointMovement


class MComCore(gym.Env):
    NOOP_ACTION = 0

    def __init__(self, stations, users, config={}):
        super().__init__()
        # set unspecified parameters to default configuration
        config = {**self.default_config(), **config}
        config = self.seeding(config)

        self.width, self.height = config["width"], config["height"]
        self.seed = config["seed"]
        self.reset_rng_episode = config["reset_rng_episode"]
        # set up arrival pattern, channel, movement, etc.
        self.arrival = config["arrival"](**config["arrival_params"])
        self.channel = config["channel"](**config["channel_params"])
        self.scheduler = config["scheduler"](**config["scheduler_params"])
        self.movement = config["movement"](**config["movement_params"])
        self.utility = config["utility"](**config["utility_params"])

        # define parameters that track the simulation's progress
        self.EP_MAX_TIME = config["EP_MAX_TIME"]
        self.time = None
        self.closed = False

        # defines the simulation's overall basestations and UEs
        self.stations = {bs.bs_id: bs for bs in stations}
        self.users = {ue.ue_id: ue for ue in users}
        self.NUM_STATIONS = len(self.stations)
        self.NUM_USERS = len(self.users)

        # define sizes of base feature set that can or cannot be observed
        self.feature_sizes = {
            "connections": self.NUM_STATIONS,
            "snrs": self.NUM_STATIONS,
            "utility": 1,
            "bcast": self.NUM_STATIONS,
            "stations_connected": self.NUM_STATIONS,
        }

        # set object that handles calls to action(), reward() & observation()
        # set action & observation space according to handler
        self.handler = config["handler"]
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

        # stores what UEs are currently active, i.e., request service
        self.active: List[UserEquipment] = None
        # stores what downlink connections between BSs and UEs are active
        self.connections: Dict[BaseStation, Set[UserEquipment]] = None
        # stores datarate of downlink connections between UEs and BSs
        self.datarates: Dict[Tuple[BaseStation, UserEquipment], float] = None
        # stores each UE's (scaled) utility
        self.utilities: Dict[UserEquipment, float] = None
        # define RNG (as of now: unused)
        self.rng = None

        # parameters for pygame visualization
        self.window = None
        self.clock = None
        self.conn_isolines = None
        self.mb_isolines = None

        # tracks performance metrics for the on-going simulation
        self.metrics = None

    @classmethod
    def default_config(cls):
        """Set default configuration of environment dynamics."""
        # set up configuration of environment
        width, height = 200, 200
        ep_time = 250
        config = {
            # environment parameters:
            "width": width,
            "height": height,
            "EP_MAX_TIME": ep_time,
            "seed": 0,
            "reset_rng_episode": False,
            # used simulation models:
            "arrival": NoDeparture,
            "channel": OkumuraHata,
            "scheduler": ResourceFair,
            "movement": RandomWaypointMovement,
            "utility": BoundedLogUtility,
            "Handler": MComCentralHandler,
        }

        # set up default configuration parameters for arrival pattern, ...
        aparams = {"ep_time": ep_time, "reset_rng_episode": False}
        config.update({"arrival_params": aparams})
        config.update({"channel_params": {}})
        config.update({"scheduler_params": {}})
        mparams = {"width": width, "height": height,
                   "reset_rng_episode": False}
        config.update({"movement_params": mparams})
        uparams = {"lower": -20, "upper": 20, "coeffs": (10, 0, 10)}
        config.update({"utility_params": uparams})

        return config

    @classmethod
    def seeding(cls, config):
        seed = config["seed"]
        keys = [
            "arrival_params",
            "channel_params",
            "scheduler_params",
            "movement_params",
            "utility_params",
        ]
        for num, key in enumerate(keys):
            config[key]["seed"] = seed + num + 1

        return config

    def reset(self):
        """Reset environment to starting state."""
        # reset time & performance metrics
        self.time = 0.0
        self.metrics = defaultdict(list)

        # initialize RNG or reset (if necessary on episode end)
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # reset state kept by arrival pattern, channel, scheduler, etc.
        self.arrival.reset()
        self.channel.reset()
        self.scheduler.reset()
        self.movement.reset()
        self.utility.reset()

        # generate new arrival and exit times for UEs
        for ue in self.users.values():
            ue.stime = self.arrival.arrival(ue)
            ue.extime = self.arrival.departure(ue)

        # generate new initial positons of UEs
        for ue in self.users.values():
            ue.x, ue.y = self.movement.initial_position(ue)

        # initially not all UEs request downlink connections (service)
        self.active = [ue for ue in self.users.values() if ue.stime <= 0]
        self.active = sorted(self.active, key=lambda ue: ue.ue_id)

        # reset established downlink connections (default empty set)
        self.connections = defaultdict(set)
        # reset connections' data rates (defaults set to 0.0)
        self.datarates = defaultdict(float)
        # reset UEs' utilities
        self.utilities = {}

        # set time of last UE's departure
        self.max_departure = max(ue.extime for ue in self.users.values())

        # check if handler is applicable to mobile scenario
        # NOTE: e.g. fails if the central handler is used,
        # although the number of UEs changes
        self.handler.check(self)

        return self.handler.observation(self)

    def apply_action(self, action: int, ue: UserEquipment) -> None:
        """Connect or disconnect `ue` to/from basestation `action`."""
        # do not apply update to connections if NOOP_ACTION is selected
        if action == self.NOOP_ACTION or ue not in self.active:
            return

        bs = self.stations[action - 1]
        # disconnect to basestation if user equipment already connected
        if ue in self.connections[bs]:
            self.connections[bs].remove(ue)

        # establish connection if user equipment not connected but reachable
        elif self.check_connectivity(bs, ue):
            self.connections[bs].add(ue)

    def check_connectivity(self, bs: BaseStation, ue: UserEquipment) -> bool:
        """Connection can be established if SNR exceeds threshold of UE."""
        snr = self.channel.snr(bs, ue)
        return snr > ue.snr_threshold

    def available_connections(self, ue: UserEquipment) -> Dict:
        """Returns dict of what basestations users could connect to."""
        stations = self.stations.values()
        return {bs for bs in stations if self.check_connectivity(bs, ue)}

    def update_connections(self) -> None:
        """Release connections where BS and UE moved out-of-range."""
        connections = {
            bs: set(ue for ue in ues if self.check_connectivity(bs, ue))
            for bs, ues in self.connections.items()
        }
        self.connections.clear()
        self.connections.update(connections)

    def step(self, actions: Dict[int, int]):
        assert not self.done, "step() called on already terminated episode"

        # apply handler to transform actions to expected shape
        actions = self.handler.action(self, actions)

        # release established connections that moved e.g. out-of-range
        self.update_connections()

        # TODO: add penalties for changing connections?
        for ue_id, action in actions.items():
            self.apply_action(action, self.users[ue_id])

        # track total number of (BS, UE) connections
        num_connections = sum([len(con) for con in self.connections.values()])
        self.metrics["num_connections"].append(num_connections)
        # track number of UEs at least connected once
        ues_connected = len(set.union(set(), *self.connections.values()))
        self.metrics["ues_connected"].append(ues_connected)

        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)
        datarates = list(self.macro.values())
        mean_datarate = np.mean(datarates) if self.macro else 0.0
        self.metrics["mean_datarate"].append(mean_datarate)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {
            ue: self.utility.utility(self.macro[ue]) for ue in self.active
        }

        # track the average utility of UEs over time
        # set to lower bound if no connections are active
        mean_utility = (
            np.mean(list(self.utilities.values()))
            if self.utilities
            else self.utility.lower
        )
        self.metrics["mean_utility"].append(mean_utility)

        # scale utilities to range [-1, 1] before computing rewards
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # terminate existing connections for exiting UEs
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.connections.items():
            self.connections[bs] = ues - leaving

        # update list of active UEs & add those that begin to request service
        self.active = sorted(
            [
                ue
                for ue in self.users.values()
                if ue.extime > self.time and ue.stime <= self.time
            ],
            key=lambda ue: ue.ue_id,
        )

        # update the data rate of each (BS, UE) connection after movement
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update internal time of environment
        self.time += 1
        # check whether episode is done & close the environment
        if self.done and self.window:
            self.close()

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.done:
            return self.step({})

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        return observation, rewards, self.done, info

    @property
    def done(self):
        """Episode is done after max. time steps or once last UE departed."""
        return self.time >= min(self.EP_MAX_TIME, self.max_departure)

    def macro_datarates(self, datarates):
        """Compute aggregated UE data rates given all its connections."""
        ue_datarates = Counter()
        for (bs, ue), datarate in self.datarates.items():
            ue_datarates.update({ue: datarate})
        return ue_datarates

    def station_allocation(self, bs) -> Dict:
        """Schedule BS's resources (e.g. phy. res. blocks) to connected UEs."""
        conns = self.connections[bs]

        # compute SNR & max. data rate for each connected user equipment
        snrs = [self.channel.snr(bs, ue) for ue in conns]

        # UE's max. data rate achievable when BS schedules all resources to it
        max_allocation = [
            self.channel.datarate(bs, ue, snr) for snr, ue in zip(snrs, conns)
        ]

        # BS shares resources among connected user equipments
        rates = self.scheduler.share(bs, max_allocation)

        return {(bs, ue): rate for ue, rate in zip(conns, rates)}

    def station_utilities(self) -> Dict[BaseStation, UserEquipment]:
        """Compute average utility of UEs connected to the basestation."""
        # set utility of BS with no active connections (idle BS) to
        # (scaled) lower utility bound
        idle = self.utility.scale(self.utility.lower)

        util = {
            bs: sum(self.utilities[ue] for ue in self.connections[bs])
            / len(self.connections[bs])
            if self.connections[bs]
            else idle
            for bs in self.stations.values()
        }

        return util

    def bs_isolines(self, drate):
        """Isolines where UEs could still receive `drate` max. data rate."""
        isolines = {}
        config = self.default_config()["ue"]

        for bs in self.stations.values():
            isolines[bs] = self.channel.isoline(
                bs, config, (self.width, self.height), drate
            )

        return isolines

    def features(self) -> Dict[int, Dict[str, np.ndarray]]:
        # fix ordering of BSs for observations
        stations = sorted(
            [bs for bs in self.stations.values()], key=lambda bs: bs.bs_id
        )

        # compute average utility of each basestation's connections
        bs_utilities = self.station_utilities()

        def ue_features(ue):
            """Define local observation vector for UEs."""
            # (1) observation of current connections
            # encodes UE's connections as one-hot vector
            connections = [bs for bs in stations if ue in self.connections[bs]]
            onehot = np.zeros(self.NUM_STATIONS, dtype=np.float32)
            onehot[[bs.bs_id for bs in connections]] = 1

            # (2) (normalized) SNR between UE to each BS
            snrs = [self.channel.snr(bs, ue) for bs in stations]
            maxsnr = max(snrs)
            snrs = np.asarray([snr / maxsnr for snr in snrs], dtype=np.float32)

            # (3) include normalized utility of UE
            utility = (
                self.utilities[ue]
                if ue in self.utilities
                else self.utility.scale(self.utility.lower)
            )
            utility = np.asarray([utility], dtype=np.float32)

            # (4) receive broadcast of average BS utilities of BSs in range
            # if broadcast is not received, set utility to lower bound
            idle = self.utility.scale(self.utility.lower)
            util_bcast = {
                bs: util if self.check_connectivity(bs, ue) else idle
                for bs, util in bs_utilities.items()
            }
            util_bcast = np.asarray(
                [util_bcast[bs] for bs in stations], dtype=np.float32
            )

            # (5) receive broadcast of (normalized) connected UE count
            # if broadcast is not received, set UE connection count to zero
            def num_connected(bs):
                if self.check_connectivity(bs, ue):
                    return len(self.connections[bs])
                return 0.0
            stations_connected = [num_connected(bs) for bs in stations]

            # normalize by the max. number of connections
            total = max(1, sum(stations_connected))
            stations_connected = np.asarray(
                [num / total for num in stations_connected], dtype=np.float32
            )

            return {
                "connections": onehot,
                "snrs": snrs,
                "utility": utility,
                "bcast": util_bcast,
                "stations_connected": stations_connected,
            }

        def dummy_features(ue):
            """Define dummy observation for non-active UEs."""
            onehot = np.zeros(self.NUM_STATIONS, dtype=np.float32)
            snrs = np.zeros(self.NUM_STATIONS, dtype=np.float32)
            utility = np.asarray(
                [self.utility.scale(self.utility.lower)], dtype=np.float32
            )
            idle = self.utility.scale(self.utility.lower)
            util_bcast = idle * np.ones(self.NUM_STATIONS, dtype=np.float32)
            num_connected = np.ones(self.NUM_STATIONS, dtype=np.float32)

            return {
                "connections": onehot,
                "snrs": snrs,
                "utility": utility,
                "bcast": util_bcast,
                "stations_connected": num_connected,
            }

        # define dummy observations for non-active UEs
        idle_ues = set(self.users.values()) - set(self.active)
        obs = {ue.ue_id: dummy_features(ue) for ue in idle_ues}
        obs.update({ue.ue_id: ue_features(ue) for ue in self.active})

        return obs

    def render(self, mode="human") -> None:
        # do not continue rendering once environment has been closed
        if self.closed:
            return

        # calculate isoline contours for BSs' connectivity range
        if self.conn_isolines is None:
            self.conn_isolines = self.bs_isolines(0.0)
        # calculate isoline contours for BSs' 1 MB/s range
        if self.mb_isolines is None:
            self.mb_isolines = self.bs_isolines(1.0)

        # set up matplotlib figure & axis configuration
        fig = plt.figure()
        fx = max(3.0 / 2.0 * 1.25 * self.width / fig.dpi, 8.0)
        fy = max(1.25 * self.height / fig.dpi, 5.0)
        plt.close()
        fig = plt.figure(figsize=(fx, fy))
        gs = fig.add_gridspec(
            ncols=2,
            nrows=3,
            width_ratios=(4, 2),
            height_ratios=(2, 3, 3),
            hspace=0.45,
            wspace=0.2,
            top=0.95,
            bottom=0.15,
            left=0.025,
            right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        dash_ax = fig.add_subplot(gs[0, 1])
        qoe_ax = fig.add_subplot(gs[1, 1])
        conn_ax = fig.add_subplot(
            gs[2, 1],
        )

        # render simulation, metrics and score if step() was called
        # i.e. this prevents rendering in the sequential environment before
        # the first round-robin of actions is finalized
        if self.time > 0:
            self.render_simulation(sim_ax)
            self.render_dashboard(dash_ax)
            self.render_mean_utility(qoe_ax)
            self.render_ues_connected(conn_ax)

        # align plots' y-axis labels
        fig.align_ylabels((qoe_ax, conn_ax))
        canvas = FigureCanvas(fig)
        canvas.draw()

        # prevents opening multiple figures on consecutive render() calls
        plt.close()

        if mode == "rgb_array":
            # render RGB image for e.g. video recording
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            # reshape image from 1d array to 2d array
            return data.reshape(canvas.get_width_height()[::-1] + (3,))

        elif mode == "human":
            # render RGBA image on pygame surface
            data = canvas.buffer_rgba()
            size = canvas.get_width_height()

            # set up pygame window to display matplotlib figure
            if self.window is None:
                pygame.init()
                self.clock = pygame.time.Clock()

                # set window size to figure's size in pixels
                window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
                self.window = pygame.display.set_mode(window_size)

                # remove pygame icon from window; set icon to empty surface
                pygame.display.set_icon(Surface((0, 0)))

                # set window's caption and background color
                pygame.display.set_caption("MComEnv")

            # clear surface
            self.window.fill("white")

            # plot matplotlib's RGBA frame on the pygame surface
            screen = pygame.display.get_surface()
            plot = pygame.image.frombuffer(data, size, "RGBA")
            screen.blit(plot, (0, 0))

            # update the full display surface to the window
            pygame.display.flip()

            # handle pygame events (such as closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        else:
            raise ValueError("Invalid rendering mode.")

    def render_simulation(self, ax) -> None:
        colormap = cm.get_cmap("RdYlGn")
        # define normalization for unscaled utilities
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)

        for ue, utility in self.utilities.items():
            # plot UE by its (unscaled) utility
            utility = self.utility.unscale(utility)
            color = colormap(unorm(utility))

            ax.scatter(ue.point.x, ue.point.y,
                       s=200, zorder=2, color=color, marker='o')
            ax.annotate(ue.ue_id, xy=(ue.point.x, ue.point.y),
                        ha='center', va='center')

        for bs in self.stations.values():
            # plot BS symbol and annonate by its BS ID
            ax.plot(
                bs.point.x,
                bs.point.y,
                marker=BS_SYMBOL,
                markersize=30,
                markeredgewidth=0.1,
                color="black",
            )
            bs_id = string.ascii_uppercase[bs.bs_id]
            ax.annotate(
                bs_id,
                xy=(bs.point.x, bs.point.y),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )

            # plot BS ranges where UEs may connect or can receive at most 1MB/s
            ax.scatter(*self.conn_isolines[bs], color="gray", s=3)
            ax.scatter(*self.mb_isolines[bs], color="black", s=3)

        for bs in self.stations.values():
            for ue in self.connections[bs]:
                # color is connection's contribution to the UE's total utility
                share = self.datarates[(bs, ue)] / self.macro[ue]
                share = share * self.utility.unscale(self.utilities[ue])
                color = colormap(unorm(share))

                # add black background/borders for lines for visibility
                ax.plot(
                    [ue.point.x, bs.point.x],
                    [ue.point.y, bs.point.y],
                    color=color,
                    path_effects=[
                        pe.SimpleLineShadow(shadow_color="black"),
                        pe.Normal(),
                    ],
                    linewidth=3,
                    zorder=-1,
                )

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])

    def render_dashboard(self, ax) -> None:

        mean_utility = self.metrics["mean_utility"][-1]
        total_mean_utility = np.mean(self.metrics["mean_utility"])
        mean_datarate = self.metrics["mean_datarate"][-1]
        total_mean_datarate = np.mean(self.metrics["mean_datarate"])

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        rows = ["Current", "History"]
        cols = ["Avg. DR [GB/s]", "Avg. Utility"]
        text = [
            [f"{mean_datarate:.3f}", f"{mean_utility:.3f}"],
            [f"{total_mean_datarate:.3}", f"{total_mean_utility:.3f}"],
        ]

        table = ax.table(
            text,
            rowLabels=rows,
            colLabels=cols,
            cellLoc="center",
            edges="B",
            loc="upper center",
            bbox=[0.0, -0.25, 1.0, 1.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    def render_mean_utility(self, ax) -> None:
        time = np.arange(self.time)
        mean_utility = self.metrics["mean_utility"]
        ax.plot(time, mean_utility, linewidth=1, color="black")

        ax.set_ylabel("Avg. Utility")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([self.utility.lower, self.utility.upper])

    def render_ues_connected(self, ax) -> None:
        time = np.arange(self.time)
        ues_connected = self.metrics["ues_connected"]
        ax.plot(time, ues_connected, linewidth=1, color="black")

        ax.set_xlabel("Time")
        ax.set_ylabel("#Conn. UEs")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, len(self.users)])

    def close(self) -> None:
        """Closes the environment and terminates its visualization."""
        pygame.quit()
        self.window = None
        self.closed = True

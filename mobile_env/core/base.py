import string
import heapq
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict

import gym
import pygame
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.backends.backend_agg as agg
from matplotlib import cm
from pygame import Surface

from mobile_env.core.util import BS_SYMBOL
from mobile_env.core.channels import OkumuraHata
from mobile_env.core.schedules import ResourceFair
from mobile_env.core.utilities import BoundedLogUtility
from mobile_env.core.entities import BaseStation, UserEquipment
from mobile_env.core.movement import RandomWaypointMovement


class MComEnv(gym.Env):
    IGNORE_ACTION = 0

    def __init__(self, stations, ues, config=None) -> None:
        super().__init__()
        if config is None:
            config = self.default_config

        self.width, self.height = config['width'], config['height']
        # set up channel, movement, etc. according to specified parameters
        self.channel = config['channel'](**config['channel_params'])
        self.scheduler = config['scheduler'](**config['scheduler_params'])
        self.movement = config['movement'](**config['movement_params'])
        self.utility = config['utility'](**config['utility_params'])

        # define parameters that track the simulation's progress
        self.EP_MAX_TIME = config['EP_MAX_TIME']
        self.time = None
        self.done = None

        # defines the simulation's overall basestations and UEs
        self.stations: Dict[int, BaseStation] = {
            bs.bs_id: bs for bs in stations}
        self.users: Dict[int, UserEquipment] = {ue.ue_id: ue for ue in ues}

        assert len(self.stations) > 0, 'Cannot simulate without any BS.'
        assert len(self.users) > 0, 'Cannot simulate without any UE.'

        self.NUM_STATIONS = len(self.stations)
        self.NUM_USERS = len(self.users)

        # stores what UEs are currently active, i.e., request service
        self.active: List[UserEquipment] = None
        # stores what downlink connections between BSs and UEs are active
        self.connections: Dict[BaseStation, Set[UserEquipment]] = None
        # stores datarate of downlink connections between UEs and BSs
        self.datarates: Dict[Tuple[BaseStation, UserEquipment], float] = None
        # stores each UE's (scaled) utility
        self.utilities: Dict[UserEquipment, float] = None

        # define action space for multi-agent setting
        self.action_space = gym.spaces.Dict({ue.ue_id: gym.spaces.Discrete(
            self.NUM_STATIONS + 1) for ue in self.users.values()})

        # define observation spaec for multi-agent setting
        self.observation_space = gym.spaces.Dict({ue.ue_id: gym.spaces.Box(
            low=-1, high=1, shape=(4 * self.NUM_STATIONS + 1,), dtype=np.float32) for ue in self.users.values()})

        # parameters for pygame visualization
        self.window = None
        self.clock = None
        self.conn_isolines = None
        self.mb_isolones = None

        # track performance metrics for the simulation
        self.metrics = None

        self.reset()

    @ classmethod
    def default_config(cls):
        """Set default configuration of environment dynamics."""
        # set up configuration of environment
        config = {'width': 200, 'height': 200, 'channel': OkumuraHata, 'scheduler': ResourceFair,
                  'movement': RandomWaypointMovement, 'utility': BoundedLogUtility}
        config.update({'channel_params': {}})
        config.update({'scheduler_params': {}})
        config.update(
            {'movement_params': {'width': config['width'], 'height': config['height']}})
        config.update(
            {'utility_params': {'lower': -20, 'upper': 20, 'coeffs': (10, 0, 10)}})

        # set up configuration of evaluation
        config.update({'EP_MAX_TIME': 1000})
        return config

    def reset(self):
        """Reset environment to starting state."""
        self.time = 0.0
        self.done = False

        # initially not all UEs request downlink connections (service)
        # sort active UEs by the time they exit (ascending exit time)
        self.active = sorted(
            [ue for ue in self.users.values() if ue.stime <= 0], key=lambda ue: ue.extime)

        # reset established downlink connections (default empty set)
        self.connections = defaultdict(set)
        # reset connections' data rates (defaults set to 0.0)
        self.datarates = defaultdict(float)
        # reset UEs' utilities
        self.utilities = {}

        # reset positions of UEs to their initial position
        for ue in self.users.values():
            ue.x, ue.y = ue.init_pos

        # reset simulation metrics
        self.metrics = defaultdict(list)

        return self.observations()

    def apply_action(self, action: int, ue: UserEquipment) -> None:
        """Connect or disconnect `ue` to/from basestation `action`."""
        # do not apply update to connections if IGNORE_ACTION is selected
        if action == self.IGNORE_ACTION or ue not in self.active:
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

    def available_connections(self) -> Dict:
        """Returns dict of what basestations users could connect to."""
        connectable = {ue: {bs for bs in self.stations.values(
        ) if self.check_connectivity(bs, ue)} for ue in self.users.values()}
        return connectable

    def update_connections(self) -> None:
        """Release connections where BS and UE moved out-of-range."""
        connections = {bs: set(ue for ue in ues if self.check_connectivity(
            bs, ue)) for bs, ues in self.connections.items()}
        self.connections.clear()
        self.connections.update(connections)

    def step(self, actions: Dict[int, int]):
        assert not self.done, 'step() called on already terminated episode'

        # release established connections that are out-of-range (due to movement)
        self.update_connections()

        # TODO: add penalties for changing connections?
        for ue_id, action in actions.items():
            self.apply_action(action, self.users[ue_id])

        # track total number of (BS, UE) connections
        num_connections = sum([len(con) for con in self.connections.values()])
        self.metrics['num_connections'].append(num_connections)
        # track number of UEs at least connected once
        ues_connected = len(set.union(set(), *self.connections.values()))
        self.metrics['ues_connected'].append(ues_connected)

        # update each (BS, UE) connection's data rate after connection re-assignment
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)
        mean_datarate = np.mean(
            list(self.macro.values())) if self.macro else 0.0
        self.metrics['mean_datarate'].append(mean_datarate)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {ue: self.utility.utility(
            self.macro[ue]) for ue in self.active}
        # track the average utility of UEs; set to lower bound if no connections are active
        mean_utility = np.mean(list(self.utilities.values())
                               ) if self.utilities else self.utility.lower
        self.metrics['mean_utility'].append(mean_utility)

        #  scale utilities to range [-1, 1] before computing rewards
        self.utilities = {ue: self.utility.scale(
            utility) for ue, utility in self.utilities.items()}

        # compute rewards from utility for each UE
        rewards = self.reward()

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # terminate existing connections for exiting UEs
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.connections.items():
            self.connections[bs] = ues - leaving

        # update list of active UEs & add those that begin to request service
        self.active = [ue for ue in self.users.values() if ue.extime >
                       self.time and ue.stime <= self.time]

        # update the data rate of each (BS, UE) connection after movement
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # compute observations for next step
        observation = self.observations()

        # update time and check whether episode is done
        self.time += 1
        if self.time >= self.EP_MAX_TIME:
            self.done = True

        info = {}
        return observation, rewards, self.done, info

    def macro_datarates(self, datarates):
        """Compute aggregated UE data rates given all its simultaneous connections."""
        ue_datarates = Counter()
        for (bs, ue), datarate in self.datarates.items():
            ue_datarates.update({ue: datarate})
        return ue_datarates

    def station_allocation(self, bs) -> Dict:
        """Schedule BS's resources (e.g. phy. res. blocks) to connected UEs."""
        connected = self.connections[bs]

        # compute SNR & max. data rate for each connected user equipment
        snrs = [self.channel.snr(bs, ue) for ue in connected]

        # UE's max. data rate achievable when BS schedules all resources to it
        max_allocation = [self.channel.datarate(
            bs, ue, snr) for snr, ue in zip(snrs, connected)]

        # BS shares resources among connected user equipments
        rates = self.scheduler.share(bs, max_allocation)

        return {(bs, ue): rate for ue, rate in zip(connected, rates)}

    def info(self):
        pass

    def reward(self):
        """Define each UEs' reward as its own utility aggregated with the average utility of nearby stations."""
        # check what BS-UE connections are possible
        connectable = self.available_connections()

        # compute average utility of UEs for each BS; set to lower bound if no UEs are connected
        bs_utilities = self.station_utilities()

        def ue_utility(ue):
            # utilities are broadcasted, i.e., aggregate utilities of BSs in range
            ngbr_utility = sum(bs_utilities[bs] for bs in connectable[ue])

            # calculate rewards as average weighted by the number of each BSs' connections
            ngbr_counts = sum(len(self.connections[bs])
                              for bs in connectable[ue])

            return (ngbr_utility + self.utilities[ue]) / (ngbr_counts + 1)

        rewards = {ue.ue_id: ue_utility(ue) for ue in self.active}
        return rewards

    def station_utilities(self) -> Dict[BaseStation, UserEquipment]:
        """Compute average utility of UEs connected to the basestation."""
        # set utility of BS with no active connections (idle BS) to (scaled) lower utility bound
        idle = self.utility.scale(self.utility.lower)

        util = {bs: sum(self.utilities[ue] for ue in self.connections[bs]) / len(
            self.connections[bs]) if self.connections[bs] else idle for bs in self.stations.values()}

        return util

    def bs_isolines(self, drate):
        """Compute isolones where UEs could receive at least `drate` max. data rate from BS."""
        isolines = {}
        config = self.default_config()['ue']

        for bs in self.stations.values():
            isolines[bs] = self.channel.isoline(
                bs, config, (self.width, self.height), drate)

        return isolines

    def observations(self) -> Dict[int, np.ndarray]:
        """Flatten (internal) observation vector to vector shape per UE."""
        obs = self._observation()
        # get list of keys used to represent each UE
        keys = next(iter(obs.values())).keys()

        # concatenate each UE's Dict observation (vector representation)
        return {ue_id: np.concatenate([obs_dict[key] for key in keys]) for ue_id, obs_dict in obs.items()}

    def _observation(self) -> Dict[int, Dict[str, np.ndarray]]:
        # fix ordering of BSs for observations
        stations = sorted(
            [bs for bs in self.stations.values()], key=lambda bs: bs.bs_id)

        # compute average utility of each basestation's connections
        bs_utilities = self.station_utilities()

        def ue_observations(ue):
            """Define local observation vector for UEs."""
            # (1) observation of current connections
            # encodes UE's connections as one-hot vector
            connections = [bs for bs in stations if ue in self.connections[bs]]
            onehot = np.zeros(self.NUM_STATIONS, dtype=np.float16)
            onehot[[bs.bs_id for bs in connections]] = 1

            # (2) (normalized) SNR between UE to each BS
            snrs = [self.channel.snr(bs, ue) for bs in stations]
            max_snr = max(snrs)
            snrs = np.asarray(
                [snr / max_snr for snr in snrs], dtype=np.float16)

            # (3) include normalized utility of UE
            utility = self.utilities[ue] if ue in self.utilities else self.utility.scale(
                self.utility.lower)
            utility = np.asarray([utility], dtype=np.float16)

            # (4) receive broadcast of average BS utilities of BSs in range
            # if broadcast is not received, set utility to lower bound
            idle = self.utility.scale(self.utility.lower)
            util_bcast = {bs: util if self.check_connectivity(
                bs, ue) else idle for bs, util in bs_utilities.items()}
            util_bcast = np.asarray([util_bcast[bs]
                                     for bs in stations], dtype=np.float16)

            # (5) receive broadcast of (normalized) connected UE count
            # if broadcast is not received, set UE connection count to zero
            num_connected = [len(self.connections[bs]) if self.check_connectivity(
                bs, ue) else 0.0 for bs in stations]
            total_conn = max(1, sum(num_connected))
            num_connected = np.asarray(
                [count / total_conn for count in num_connected], dtype=np.float16)

            return {'connections': onehot, 'snrs': snrs, 'utility': utility, 'bcast': util_bcast, 'stations_connected': num_connected}

        def dummy_observations(ue):
            """Define dummy observation for non-active UEs."""
            onehot = np.zeros(self.NUM_STATIONS, dtype=np.float16)
            snrs = np.zeros(self.NUM_STATIONS, dtype=np.float16)
            utility = np.asarray(
                [self.utility.scale(self.utility.lower)], dtype=np.float16)
            idle = self.utility.scale(self.utility.lower)
            util_bcast = idle * np.ones(self.NUM_STATIONS, dtype=np.float16)
            num_connected = np.ones(self.NUM_STATIONS, dtype=np.float16)

            return {'connections': onehot, 'snrs': snrs, 'utility': utility, 'bcast': util_bcast, 'stations_connected': num_connected}

        # define dummy observations for non-active UEs
        idle_ues = set(self.users.values()) - set(self.active)
        obs = {ue.ue_id: dummy_observations(ue) for ue in idle_ues}
        obs.update({ue.ue_id: ue_observations(ue) for ue in self.active})

        return obs

    def render(self, mode="human") -> None:
        # set up matplotlib figure & axis configuration
        fig = plt.figure(figsize=(7.5, 3.7))
        gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=(3, 2), height_ratios=(
            2, 3, 3), hspace=0.45, wspace=0.2, top=0.95, bottom=0.15, left=0.025, right=0.955)

        sim_ax = fig.add_subplot(gs[:, 0])
        dash_ax = fig.add_subplot(gs[0, 1])
        conn_ax = fig.add_subplot(gs[2, 1], )
        qoe_ax = fig.add_subplot(gs[1, 1])

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

            # calculate isoline contours for BSs' connectivity range
            self.conn_isolines = self.bs_isolines(0.0)
            self.mb_isolines = self.bs_isolines(1.0)

        # clear surface
        self.window.fill("white")

        # render simulation, metrics and score
        self.render_simulation(sim_ax)
        self.render_dashboard(dash_ax)
        self.render_mean_utility(qoe_ax)
        self.render_ues_connected(conn_ax)

        # align plots' y-axis labels
        fig.align_ylabels((qoe_ax, conn_ax))

        # render matplotlib plot as RGB frame on canvas
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        data = canvas.buffer_rgba()
        size = canvas.get_width_height()

        # plot matplotlib's RGBA frame on the pygame surface
        screen = pygame.display.get_surface()
        plot = pygame.image.frombuffer(data, size, "RGBA")
        screen.blit(plot, (0, 0))
        plt.close()

        # update the full display surface to the window
        pygame.display.flip()

        # handle pygame events (such as closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def render_simulation(self, ax) -> None:
        colormap = cm.get_cmap('RdYlGn')
        # define normalization for unscaled utilities
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)

        for ue, utility in self.utilities.items():
            # plot UE by its (unscaled) utility
            utility = self.utility.unscale(utility)
            color = colormap(unorm(utility))

            ax.scatter(ue.point.x, ue.point.y, s=200,
                       zorder=2, color=color, marker='o')
            ax.annotate(ue.ue_id, xy=(ue.point.x, ue.point.y),
                        ha='center', va='center')

        for bs in self.stations.values():
            # plot BS symbol and annonate by its BS ID
            ax.plot(bs.point.x, bs.point.y, marker=BS_SYMBOL,
                    markersize=30, markeredgewidth=0.1, color='black')
            bs_id = string.ascii_uppercase[bs.bs_id]
            ax.annotate(bs_id, xy=(bs.point.x, bs.point.y), xytext=(
                0, -25), ha='center', va='bottom', textcoords='offset points')

            # plot ranges where connections to the BS are possible or yield 1 MB/s
            ax.scatter(*self.conn_isolines[bs], color='gray', s=3)
            ax.scatter(*self.mb_isolines[bs], color='black', s=3)

        for bs in self.stations.values():
            for ue in self.connections[bs]:
                # plot connection's color dependend on its contribution to the UE's total utility
                share = self.datarates[(bs, ue)] / self.macro[ue]
                # weight share of connection's data rate of macro data rate by UE's utility
                share = share * self.utility.unscale(self.utilities[ue])
                color = colormap(unorm(share))

                # add black background/borders for lines to make them better visible if the utility color is too light
                ax.plot([ue.point.x, bs.point.x], [ue.point.y, bs.point.y], color=color, path_effects=[
                        pe.SimpleLineShadow(shadow_color='black'), pe.Normal()], linewidth=3, zorder=-1)

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])

    def render_dashboard(self, ax) -> None:

        mean_utility = self.metrics['mean_utility'][-1]
        total_mean_utility = np.mean(self.metrics['mean_utility'])
        mean_datarate = self.metrics['mean_datarate'][-1]
        total_mean_datarate = np.mean(self.metrics['mean_datarate'])

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        rows = ['Current', 'History']
        cols = ['Avg. DR [GB/s]', 'Avg. Utility']
        text = [[f'{mean_datarate:.3f}', f'{mean_utility:.3f}'],
                [f'{total_mean_datarate:.3}', f'{total_mean_utility:.3f}']]

        table = ax.table(text, rowLabels=rows, colLabels=cols, cellLoc='center',
                         edges='B', loc='upper center', bbox=[0.0, -0.25, 1.0, 1.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    def render_mean_utility(self, ax) -> None:
        time = np.arange(self.time)
        mean_utility = self.metrics['mean_utility']
        ax.plot(time, mean_utility, linewidth=1, color='black')

        ax.set_ylabel('Avg. Utility')
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([self.utility.lower, self.utility.upper])

    def render_ues_connected(self, ax) -> None:
        time = np.arange(self.time)
        ues_connected = self.metrics['ues_connected']
        ax.plot(time, ues_connected, linewidth=1, color='black')

        ax.set_xlabel('Time')
        ax.set_ylabel('#Conn. UEs')
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, len(self.users)])

    def close(self) -> None:
        self.done = True
        pygame.quit()
        self.window = None

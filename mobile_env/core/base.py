import string
import heapq
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict

import gym
import pygame
import matplotlib
#
# matplotlib.use("Agg")
#
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
from mobile_env.core.rewards import MultiUserReward


class MComEnv(gym.Env):
    IGNORE_ACTION = 0

    def __init__(self, config=None, stations=None, ues=None) -> None:
        super().__init__()
        if config is None:
            config = self.default_config

        self.width, self.height = config['width'], config['height']
        # set up channel, movement, etc. according to specified parameters
        self.channel = config['channel'](**config['channel_params'])
        self.scheduler = config['scheduler'](**config['scheduler_params'])
        self.movement = config['movement'](**config['movement_params'])
        self.utility = config['utility'](**config['utility_params'])
        self.reward = config['reward'](**config['reward_params'])

        # define parameters that track the simulation's progress
        self.EP_MAX_TIME = config['EP_MAX_TIME']
        self.time = None
        self.done = None

        # defines the simulation's overall basestations and UEs
        self.stations: Dict[int, BaseStation] = {
            bs.bs_id: bs for bs in stations}
        self.users: Dict[int, UserEquipment] = {ue.ue_id: ue for ue in ues}

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

        # TODO: define observation spaec for multi-agent setting

        # parameters for pygame visualization
        self.window_width, self.window_height = 600, 400
        self.hzoom = self.window_height / self.height
        self.wzoom = self.window_width / self.width
        self.window = None
        self.clock = None

        # track performance metrics for the simulation
        self.metrics = None

        self.reset()

    @classmethod
    def default_config(cls):
        # set up configuration of environment
        config = {'width': 100, 'height': 100, 'channel': OkumuraHata, 'scheduler': ResourceFair,
                  'movement': RandomWaypointMovement, 'utility': BoundedLogUtility, 'reward': MultiUserReward}
        config.update({'channel_params': {}})
        config.update({'scheduler_params': {}})
        config.update(
            {'movement_params': {'width': config['width'], 'height': config['height']}})
        config.update(
            {'utility_params': {'lower': -20, 'upper': 20, 'coeffs': (10, 0, 10)}})
        config.update({'reward_params': {}})

        # set up configuration of evaluation
        config.update({'EP_MAX_TIME': 100})
        return config

    def reset(self):
        self.time = 0.0
        self.done = False

        # TODO: not possible anymore!!
        # sort UEs by the time they begin to request service
        # self.users = sorted(self.users, key=lambda ue: ue.stime)

        # initially not all UEs request downlink connections (service)
        # sort active UEs by the time they exit (ascending exit time)
        self.active = sorted(
            [ue for ue in self.users.values() if ue.stime <= 0])

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

        # update the datarate of each (BS, UE) connection
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # track the average data rate of UEs
        mean_datarate = np.mean(list(self.datarates.values()))
        self.metrics['mean_datarate'].append(mean_datarate)

        # compute macro datarates for each UE & log its mean value
        macro_datarates = self.macro_datarates(self.datarates)
        mean_macro_datarate = np.mean(list(macro_datarates.values()))
        self.metrics['mean_macro_datarates'].append(mean_macro_datarate)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {ue: self.utility.utility(
            macro_datarates[ue]) for ue in self.active}
        # track the average utility of UEs
        mean_utility = np.mean(list(self.utilities.values()))
        self.metrics['mean_utility'].append(mean_utility)

        #  scale utilities to range [-1, 1] before computing rewards
        self.utilities = {ue: self.utility.scale(
            utility) for ue, utility in self.utilities.items()}

        # compute rewards from utility for each UE
        connectable = self.available_connections()
        rewards = self.reward.rewards(
            self.utilities, self.connections, connectable)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # TODO: what is the interface for handling added / removed agents!?
        # identify user equipments that left the map
        #left = [ue for ue in self.users if self.check_left_map(ue)]
        # terminate all their remaining connections

        # (4) add new UEs; remove UEs that leave map (?)

        # update the data rate of each (BS, UE) connection after movement
        # values must be updated before computing observations
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # compute observations for next step
        observation = None

        # update internal metrics

        # (7) update time

        # update metrics
        self.time += 1

        if self.time >= self.EP_MAX_TIME:
            self.done = True

        info = {}
        return self.observations, rewards, self.done, info

    def macro_datarates(self, datarates):
        ue_datarates = Counter()
        for (bs, ue), datarate in self.datarates.items():
            ue_datarates.update({ue: datarate})
        return ue_datarates

    def station_allocation(self, bs) -> Dict:
        connected = self.connections[bs]

        # compute SNR & max. data rate for each connected user equipment
        snrs = [self.channel.snr(bs, ue) for ue in connected]

        # UE's max. data rate achievable when BS schedules all resources to it
        max_allocation = [bs.bw * np.log2(1 + snr) for snr in snrs]

        # BS shares resources among connected user equipments
        rates = self.scheduler.share(bs, max_allocation)

        return {(bs, ue): rate for ue, rate in zip(connected, rates)}

    def info(self):
        pass

    def reward(self, utilities):
        pass

    def observations(self):
        # (1) observation of current connections
        # get connections of each UE as mapping UE -> BSs
        connections = defaultdict(list)
        for bs, ues in self.connections.items():
            for ue in ues:
                connections[ue].append(bs)

        # encode each UE's connections as one-hot vector
        conn_obs = {}
        for ue, bss in connections.items():
            onehot = np.zeros(self.NUM_USERS)
            onehot[bss] = 1
            conn_obs[ue] = onehot

    def render(self, mode="human"):
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
        # define colormap for unscaled utilities
        colormap = cm.get_cmap('RdYlGn')
        norm = plt.Normalize(self.utility.lower, self.utility.upper)

        for ue, utility in self.utilities.items():
            # plot UE by its (unscaled) utility
            utility = self.utility.unscale(utility)
            color = colormap(norm(utility))

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
            # TODO: how to get these ranges?
            range_conn = bs.point.buffer(69)
            range_1mbit = bs.point.buffer(46)
            ax.plot(*range_1mbit.exterior.xy, color='black')
            ax.plot(*range_conn.exterior.xy, color='gray')

        for bs in self.stations.values():
            for ue in self.connections[bs]:
                # plot connection's color dependend on its utility
                drate = self.datarates[(bs, ue)]
                utilty = self.utility.utility(drate)
                color = colormap(norm(utilty))

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

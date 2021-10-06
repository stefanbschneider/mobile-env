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
        self.basestations: List[BaseStation] = stations
        self.users: List[UserEquipment] = ues

        # stores what UEs are currently active, i.e., request service
        self.active: List[UserEquipment] = None
        # stores what downlink connections between BSs and UEs are active
        self.connections: Dict[BaseStation, Set[UserEquipment]] = None
        # stores datarate of downlink connections between UEs and BSs
        self.datarates: Dict[Tuple[BaseStation, UserEquipment], float] = None

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

        # sort UEs by the time they begin to request service
        self.users = sorted(self.users, key=lambda ue: ue.stime)

        # initially not all UEs request downlink connections (service)
        # sort active UEs by the time they exit (ascending exit time)
        self.active = sorted([ue for ue in self.users if ue.stime <= 0])

        # reset established downlink connections (default empty set)
        self.connections = defaultdict(set)
        # reset connections' data rates (default set to 0.0)
        self.datarates = defaultdict(float)

        # reset positions of UEs to their initial position
        for ue in self.users:
            ue.reset()

        # reset simulation metrics
        self.metrics = defaultdict(list)

        return self.observations()

    def close(self) -> None:
        self.done = True
        pygame.quit()
        self.window = None

    def observations(self):
        pass

    def info(self):
        return {'time': self.time}

    def update_connection(self, action: int, ue: UserEquipment):
        # do not apply update to connections if IGNORE_ACTION is selected
        if action == self.IGNORE_ACTION:
            return

        bs = self.basestations[action - 1]
        # disconnect to basestation if user equipment already connected
        if ue in self.connections[bs]:
            self.connections[bs].remove(ue)

        # establish connection if user equipment not connected but reachable
        elif self.check_connectivity(bs, ue):
            self.connections[bs].add(ue)

    def check_connectivity(self, bs, ue):
        """Connection can be established if SNR exceeds threshold of UE."""
        snr = self.channel.snr(bs, ue)
        return snr > ue.snr_threshold

    def available_connections(self):
        """Returns dict of what basestations users could connect to."""
        connectable = {ue: {bs for bs in self.basestations if self.check_connectivity(
            bs, ue)} for ue in self.users}
        return connectable

    def step(self, actions: Tuple[int, ...]):
        assert not self.done, 'step() called on already terminated episode'

        # TODO: how to handle missing actions? just ignore? SO FAR: YES
        # TODO: add penalties for changing connections?
        for action, ue in zip(actions, self.active):
            self.update_connection(action, ue)

        # update connection metrics
        num_connections = sum([len(conn)
                               for conn in self.connections.values()])
        self.metrics['num_connections'].append(num_connections)
        ues_connected = len(set.union(*self.connections.values()))
        self.metrics['ues_connected'].append(ues_connected)

        # update the datarate of each (BS, UE) connection
        for bs in self.basestations:
            rates = self.station_allocation(bs)
            self.datarates.update(rates)

        # compute macro datarates for each UE & log its mean value
        macro_datarates = self.macro_datarates(self.datarates)
        mean_macro_datarate = np.mean(list(macro_datarates.values()))
        self.metrics['mean_macro_datarates'].append(mean_macro_datarate)

        # compute utilities from UEs' data rates & log its mean value
        utilities = {ue: self.utility.utility(
            macro_datarates[ue]) for ue in self.active}
        mean_utility = np.mean(list(utilities.values()))
        self.metrics['mean_utility'].append(mean_utility)

        # compute rewards from utility for each UE
        connectable = self.available_connections()
        rewards = self.reward.rewards(utilities, self.connections, connectable)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # TODO: what is the interface for handling added / removed agents!?
        # identify user equipments that left the map
        #left = [ue for ue in self.users if self.check_left_map(ue)]
        # terminate all their remaining connections

        # (4) add new UEs; remove UEs that leave map (?)

        # update the datarate of each (BS, UE) connection after movement
        for bs in self.basestations:
            rates = self.station_allocation(bs)
            self.datarates.update(rates)

        # compute observations for next step
        observation = None

        # update internal metrics

        # (7) update time

        # update metrics
        self.time += 1
        self.metrics['avg_qoe'].append(2.0)
        self.metrics['avg_drate'].append(5.0)
        self.metrics['num_connected'].append(3.0)

        if self.time >= self.EP_MAX_TIME:
            self.done = True

        rewards = None

        info = {}
        return self.observations, rewards, self.done, info

    def macro_datarates(self, datarates):
        ue_datarates = Counter()
        for (bs, ue), datarate in self.datarates.items():
            ue_datarates.update({ue: datarate})
        return ue_datarates

    def render(self, mode="human"):
        # set up matplotlib figure & axis configuration
        fig = plt.figure(figsize=(7.5, 4))
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
        radius = 2

        colormap = cm.get_cmap('RdYlGn')
        norm = plt.Normalize(-20, 20)
        ue_macro = self.macro_datarates(self.datarates)
        # ue_utilities =

        # plot UEs
        for ue in self.active:
            # TODO: UTILITY!
            utility = ue_macro[ue]
            color = colormap(norm(utility))

            ax.plot(*ue.point.buffer(radius).exterior.xy,
                    color=color, marker="o", markersize=8)
            ax.annotate(ue.ue_id, xy=(ue.point.x, ue.point.y),
                        ha='center', va='center')

        for bs in self.basestations:
            # plot BS symbol and annonate by its BS ID
            ax.plot(bs.point.x, bs.point.y, marker=BS_SYMBOL,
                    markersize=30, markeredgewidth=0.1, color='black')
            bs_id = string.ascii_uppercase[bs.bs_id]
            ax.annotate(bs_id, xy=(bs.point.x, bs.point.y), xytext=(
                0, -25), ha='center', va='bottom', textcoords='offset points')

            # plot ranges where connections to the BS are possible or yield 1 MB/s
            range_conn = bs.point.buffer(69)
            range_1mbit = bs.point.buffer(46)
            ax.plot(*range_1mbit.exterior.xy, color='black')
            ax.plot(*range_conn.exterior.xy, color='gray')

        # plot BS-UE connections in terms of their QoE
        for bs in self.basestations:
            for ue in self.connections[bs]:
                rate = self.datarates[(bs, ue)]
                color = colormap(norm(rate))

            # add black background/borders for lines to make them better visible if the utility color is too light
            ax.plot([ue.point.x, bs.point.x], [ue.point.y, bs.point.y], color=color,
                    path_effects=[pe.SimpleLineShadow(shadow_color='black'),
                                  pe.Normal()])

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def render_dashboard(self, ax) -> None:
        # TODO!!!!
        avg_dr = 0

        mean_utility = self.metrics['mean_utility'][-1]
        total_mean_utility = np.mean(self.metrics['mean_utility'])

        # remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        text = [
            ['Curr. Avg. Rate', f'{avg_dr:.2f} GB/s'],
            ['Total Avg. Rate', f'{avg_dr:.2f} GB/s'],
            ['Curr. Avg. Utility', f'{mean_utility:.2f}'],
            ['Total Avg. Utility', f'{total_mean_utility:.2f}']
        ]

        table = ax.table(text, cellLoc='left', edges='open',
                         loc='upper center', bbox=[0.0, -0.25, 1.0, 1.25])
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

    def station_allocation(self, bs):
        connected = self.connections[bs]

        # compute SNR & max. data rate for each connected user equipment
        snrs = [self.channel.snr(bs, ue) for ue in connected]

        # UE's max. data rate achievable when BS schedules all resources to it
        max_allocation = [bs.bw * np.log2(1 + snr) for snr in snrs]

        # BS shares resources among connected user equipments
        rates = self.scheduler.share(bs, max_allocation)

        return {(bs, ue): rate for ue, rate in zip(connected, rates)}

    @staticmethod
    def compute_reward(datarates):
        # TODO:
        return 0.0

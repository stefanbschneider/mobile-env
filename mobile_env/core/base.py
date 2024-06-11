import string
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import gymnasium
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame import Surface
import random
import logging

from mobile_env.core import metrics
from mobile_env.core.arrival import NoDeparture
from mobile_env.core.channels import OkumuraHata
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.monitoring import Monitor
from mobile_env.core.movement import RandomWaypointMovement
from mobile_env.core.schedules import ResourceFair, RateFair, ProportionalFair, RoundRobin
from mobile_env.core.util import BS_SYMBOL, SENSOR_SYMBOL, deep_dict_merge
from mobile_env.core.utilities import BoundedLogUtility
from mobile_env.handlers.central import MComCentralHandler
from mobile_env.core.buffers import Buffer, PacketGenerator


class MComCore(gymnasium.Env):
    NOOP_ACTION = 0
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, stations, users, sensors, config={}, render_mode=None):
        super().__init__()

        logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.render_mode = render_mode
        assert render_mode in self.metadata["render_modes"] + [None]

        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)
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
        self.sensors = {sensor.sensor_id: sensor for sensor in sensors}
        self.NUM_STATIONS = len(self.stations)
        self.NUM_USERS = len(self.users)
        self.NUM_SENSORS = len(self.sensors)

        # define sizes of base feature set that can or cannot be observed
        self.feature_sizes = {
            "connections": self.NUM_STATIONS,
            "snrs": self.NUM_STATIONS,
            "utility": 1,
            "bcast": self.NUM_STATIONS,
            "stations_connected": self.NUM_STATIONS,
            "sensors": self.NUM_SENSORS,
        }

        # set object that handles calls to action(), reward() & observation()
        # set action & observation space according to handler
        self.handler = config["handler"]
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

        # stores what UEs are currently active, i.e., request service
        self.active: List[UserEquipment] = None
        # stores what sensors are currently active, i.e., request service
        self.active_sensor: List[Sensor] = None
        # stores what downlink connections between BSs and UEs are active
        self.connections: Dict[BaseStation, Set[UserEquipment]] = None
        # stores what downlink connections between BSs and Sensors are active
        self.connections_sensor: Dict[BaseStation, Set[Sensor]] = None
        # stores datarate of downlink connections between UEs and BSs
        self.datarates: Dict[Tuple[BaseStation, UserEquipment], float] = None
        # stores datarate of downlink connections between UEs and BSs
        self.datarates_sensor: Dict[Tuple[BaseStation, Sensor], float] = None
        # stores each UE's (scaled) utility
        self.utilities: Dict[UserEquipment, float] = None
        # stores each Sensor's (scaled) utility
        self.utilities_sensor: Dict[Sensor, float] = None
        # define RNG (as of now: unused)
        self.rng = None

        # Initialize or update queue logs
        if not hasattr(self, 'queue_logs'):
            self.queue_logs = {
                'time': [],
                'sensor_queues': [],
                'ue_queues': [],
                'bs_queues': []
            }

        # parameters for pygame visualization
        self.window = None
        self.clock = None
        self.conn_isolines = None
        self.mb_isolines = None

        # add metrics required for visualization & set up monitor
        config["metrics"]["scalar_metrics"].update(
            {
                "number connections": metrics.number_connections,
                "number connected": metrics.number_connected,
                "mean utility": metrics.mean_utility,
                "mean datarate": metrics.mean_datarate,
                "sensor_measurements": metrics.sensor_measurements,
            }
        )
        self.monitor = Monitor(**config["metrics"])

    @classmethod
    def default_config(cls):
        """Set default configuration of environment dynamics."""
        # set up configuration of environment
        width, height = 200, 200
        ep_time = 100
        config = {
            # environment parameters:
            "width": width,
            "height": height,
            "EP_MAX_TIME": ep_time,
            "seed": 10,
            "reset_rng_episode": False,
            # used simulation models:
            "arrival": NoDeparture,
            "channel": OkumuraHata,
            "scheduler": RoundRobin,
            "movement": RandomWaypointMovement,
            "utility": BoundedLogUtility,
            "handler": MComCentralHandler,
            # default cell config
            "bs": {"bw": 20e6, "freq": 2500, "tx": 40, "height": 50},
            # default UE config
            "ue": {
                "velocity": 1.5,
                "snr_tr": 2e-8,
                "noise": 1e-9,
                "height": 1.5,
            },
            # default Sensor config
            "sensor": {
                "height": 30,
                "radius": 5,
                "snr_tr": 2e-8,
                "range": 0,
                "velocity": 0,
                "logs": {},
            }          
        }

        # set up default configuration parameters for arrival pattern, ...
        aparams = {"ep_time": ep_time, "reset_rng_episode": False}
        config.update({"arrival_params": aparams})
        config.update({"channel_params": {}})
        config.update({"scheduler_params": {"quantum": 2.0}})
        mparams = {
            "width": width,
            "height": height,
            "reset_rng_episode": False,
        }
        config.update({"movement_params": mparams})
        uparams = {"lower": -20, "upper": 20, "coeffs": (10, 0, 10)}
        config.update({"utility_params": uparams})

        # set up default configuration for tracked metrics
        config.update(
            {
                "metrics": {
                    "scalar_metrics": {},
                    "ue_metrics": {},
                    "bs_metrics": {},
                    "ss_metrics": {},
                }
            }
        )

        return config

    @classmethod
    def seeding(cls, config):
        """Return config with updated and rotated seeds."""

        seed = config["seed"]
        keys = [
            "arrival_params",
            "channel_params",
            "scheduler_params",
            "movement_params",
            "utility_params",
        ]
        for num, key in enumerate(keys):
            if key not in config:
                config[key] = {}
            config[key]["seed"] = seed + num + 1

        return config

    def reset(self, *, seed=None, options=None):
        """Reset env to starting state. Return the initial obs and info."""
        super().reset(seed=seed)

        # reset time
        self.time = 0.0
        self.current_time = 0

        # set seed
        if seed is not None:
            self.seeding({"seed": seed})

        # initialize RNG or reset (if necessary on episode end)
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # extra options currently not supported
        if options is not None:
            raise NotImplementedError(
                "Passing extra options on env.reset() is not supported."
            )

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
        # reset established downlink connections for sensors (default empty set)
        self.connections_sensor = defaultdict(set)
        # reset connections' data rates (defaults set to 0.0)
        self.datarates = defaultdict(float)
        # reset connections' data rates (defaults set to 0.0)
        self.datarates_sensor = defaultdict(float)
        # reset UEs' utilities
        self.utilities = {}

        # set time of last UE's departure
        self.max_departure = max(ue.extime for ue in self.users.values())

        # reset episode's results of metrics tracked by the monitor
        self.monitor.reset()

        # check if handler is applicable to mobile scenario
        # NOTE: e.g. fails if the central handler is used,
        # although the number of UEs changes
        self.handler.check(self)

        # info
        info = self.handler.info(self)
        # store latest monitored results in `info` dictionary
        info = {**info, **self.monitor.info()}

        # #reset the sensor's logs 
        # for sensor in self.sensors.values():
        #     sensor.logs.clear()
        
        
        return self.handler.observation(self), info

    def apply_action(self, action: int, ue: UserEquipment) -> None:
        """Connect or disconnect `ue` to/from basestation `action`."""
        # Do not apply update to connections if NOOP_ACTION is selected
        if action == self.NOOP_ACTION or ue not in self.active:
            return

        # Disconnect the UE from its current BS, if any
        for bs in self.connections:
            if ue in self.connections[bs]:
                self.connections[bs].remove(ue)
                logging.info(f"UE {ue} disconnected from BS {bs}")

        bs = self.stations[action - 1]

    # Establish connection if user equipment not connected but reachable
        if self.check_connectivity(bs, ue):
            self.connections[bs].add(ue)
            logging.info(f"UE {ue} connected to BS {bs}")
            
    def check_connectivity_ss(self, bs: BaseStation, ss: Sensor) -> bool:
        '''This function should return the connectivity between Sensors and base sstations'''
        pass
        
        
        
    def check_connectivity(self, bs: BaseStation, ue: UserEquipment) -> bool:
        """Connection can be established if SNR exceeds threshold of UE."""
        snr = self.channel.snr(bs, ue)
        return snr > ue.snr_threshold

    def available_connections(self, ue: UserEquipment) -> Set:
        """Returns set of what base stations users could connect to."""
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

    def update_positions(self):
        '''Checks if ue in range and adds the timestamp to the UE'''
        for ue in self.users.values():  # Iterates over all the UEs 
            ue_point = ue.point 

            for sensor in self.sensors.values():  # Iterate over all sensors 
                if sensor.is_within_range(ue_point):
                    # Ensure ue.ue_id is a string if sensor.logs uses string keys
                    ue_id_str = str(ue.ue_id)
                    # Check if the UE is detected, initialize a list if not
                    if ue_id_str not in sensor.logs:
                        sensor.logs[ue_id_str] = [self.current_time]
                    else:
                        # If the UE has already been detected, append the current time
                        sensor.logs[ue_id_str].append(self.current_time)

    def connect_bs_sensor(self) -> None:
        """Connect each sensor to the closest base station."""
        for sensor in self.sensors.values():
            closest_bs = None
            min_distance = float('inf')
            
            for bs in self.stations.values():
                distance = np.sqrt((sensor.x - bs.x) ** 2 + (sensor.y - bs.y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs

            if closest_bs:
                sensor.connected_base_station = closest_bs
                if closest_bs not in self.connections_sensor:
                    self.connections_sensor[closest_bs] = set()
                self.connections_sensor[closest_bs].add(sensor)

    def connect_bs_ue(self) -> None:
        """Connect the UE to the closest base station within data range."""
        for ue in self.users.values():
            closest_bs = None
            min_distance = float('inf')

            for bs in self.stations.values():
                distance = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs

            if closest_bs:
                if closest_bs not in self.connections:
                    self.connections[closest_bs] = set()
                self.connections[closest_bs].add(ue)

    def transfer_data(self) -> None:
        """Transfers data from devices to base stations according to data rates."""
        for bs, ues in self.connections.items():
            for ue in ues:
                if ue.data_buffer_uplink.current_size > 0:
                    packet = ue.data_buffer_uplink.get()

                    # Check data rate for the connection
                    data_rate = self.datarates.get((bs, ue), 1e6)
                    if data_rate <= 0:
                        logging.warning(f"No data rate for connection with base station {bs} for device {ue}. Packet transmission aborted.")
                        continue

                    # Update the start of serving time if no bits served
                    if packet['remaining_size'] == packet['initial_size']:
                        packet['serving_time_start'] = self.time
                        packet['serving_bs'] = bs.bs_id

                    # Calculate the amount of bits to send based on data rate
                    bits_to_send = min(packet['remaining_size'], data_rate)

                    # Update remaining size of the packet
                    packet['remaining_size'] -= bits_to_send

                    # Log the transmission start
                    logging.info(f"Time step: {self.time} Device: {ue}, Base Station: {bs}, index: {packet['index']} "
                                 f"Bits sent: {bits_to_send}, Remaining size: {packet['remaining_size']}")

                    # Check if all bits are sent
                    if packet['remaining_size'] <= 0:
                        # Remove packet from device queue
                        ue.data_buffer_uplink.remove()
                        # Put packet into base station queue
                        bs.data_buffer_uplink.add(packet)
                        # Update base station number, time step, etc.
                        packet['serving_time_end'] = self.time
                        packet['serving_time_total'] = packet['serving_time_end'] - packet['serving_time_start']
                        logging.info(f"Time step: {self.time} Packet {packet['index']} transferred from device {ue} to base station {bs} with serving time {packet['serving_time_total']}.")

                    else:
                        logging.info(f"Time step: {self.time} Packet {packet['index']} partially transferred from device {ue} to base station {bs}.")
    
    def packet_generation(self, ue: UserEquipment) -> None:
        """Generate packets for user equipments at each time step according to Bernoulli distribution."""
        if random.random() < 0.7:  # Example probability for packet generation
            packet = PacketGenerator.create_packet(self.time)
            if ue.data_buffer_uplink.add(packet):
                logging.info(f"Time step: {self.time} Packet generated: {packet['index']} at time: {packet['creation_time']} by device {ue} with initial size of {packet['initial_size']} and remaining size of {packet['remaining_size']}")

    def packet_generation_sensor(self, sensor: Sensor) -> None:
        """Generate packets for sensors at each time step for environmental updates."""
        if random.random() < 0.1:  # Example probability for packet generation
            packet = PacketGenerator.create_packet(self.time)
            if sensor.data_buffer_uplink.add(packet):
                logging.info(f"Time step: {self.time} Packet generated: {packet['index']} at time: {packet['creation_time']} by sensor {sensor} with initial size of {packet['initial_size']} and remaining size of {packet['remaining_size']}")
    
    def log_connections(self):
        """Logs connections between bs and ues."""
        for bs, ues in self.connections.items():
            if ues:
                connected_devices = ", ".join([str(ue) for ue in ues])
                logging.info(f"Time step: {self.time} Devices {connected_devices} connected to {bs}")

    def log_datarates(self):
        """Logs datarates of each connected ue-bs pair."""
        for (bs, ue), rate in self.datarates.items():
            logging.info(f"Time step: {self.time} Data rate for {ue} connected to {bs} is : {rate}")

    def log_device_uplink_queue(self):
        """Logs the packet indexes, initial sizes, and remaining sizes for every packet in the uplink buffer of ues."""
        for ue in self.users.values():
            if ue.data_buffer_uplink.current_size > 0:
                for i in range(ue.data_buffer_uplink.current_size):
                    packet = ue.data_buffer_uplink.data_queue[i]
                    logging.info(f"Time step: {self.time} Device: {ue.ue_id}, Packet index: {packet['index']}, Initial size: {packet['initial_size']}, Remaining size: {packet['remaining_size']}")

    def log_sensor_uplink_queue(self):
        """Logs the packet indexes, initial sizes, and remaining sizes for every packet in the uplink buffer of sensors."""
        for sensor in self.sensors.values():
            if sensor.data_buffer_uplink.current_size > 0:
                for i in range(sensor.data_buffer_uplink.current_size):
                    packet = sensor.data_buffer_uplink.data_queue[i]
                    logging.info(f"Time step: {self.time} Sensor: {sensor.sensor_id}, Packet index: {packet['index']}, Initial size: {packet['initial_size']}, Remaining size: {packet['remaining_size']}")

    def log_bs_queue(self):
        """Logs the packet indexes, initial sizes, and remaining sizes for every packet in the bs queues."""
        for bs in self.stations.values():
            if bs.data_buffer_uplink:
                for i in range(bs.data_buffer_uplink.current_size):
                    packet = bs.data_buffer_uplink.data_queue[i]
                    logging.info(f"Time step: {self.time} Base station: {bs.bs_id}, Packet index: {packet['index']}, Initial size:{packet['initial_size']}, Remaining size: {packet['remaining_size']}")

    def log_all_connections(self):
        """Log all connections between base stations and user equipment in one line."""
        connection_strings = []

        for bs, ues in self.connections.items():
            sorted_ue_ids = sorted([ue.ue_id for ue in ues])
            ue_ids = ','.join(map(str, sorted_ue_ids))
            connection_strings.append(f"BS {bs.bs_id}: [{ue_ids}]")

        log_message = "Connections UEs: " + "; ".join(connection_strings)
        logging.info(log_message)

    def log_all_connections_sensors(self):
        """Log all connections between base stations and user equipment in one line."""
        connection_strings = []

        for bs, sensors in self.connections_sensor.items():
            sorted_sensor_ids = sorted([sensor.sensor_id for sensor in sensors])
            sensor_ids = ','.join(map(str, sorted_sensor_ids))
            connection_strings.append(f"BS {bs.bs_id}: [{sensor_ids}]")

        log_message = "Connections Sensors: " + "; ".join(connection_strings)
        logging.info(log_message)

    def step(self, actions: Dict[int, int]):
        assert not self.time_is_up, "step() called on terminated episode"

        # apply handler to transform actions to expected shape
        #actions = self.handler.action(self, actions)

        # release established connections that moved e.g. out-of-range
        self.update_connections()
        self.update_positions()

        # Connect sensors to the closest base station
        self.connect_bs_ue()
        self.connect_bs_sensor()

        # Logging base station connections
        logging.info(f"Time step: {self.time} Logging BS-UE connections...")
        self.log_all_connections()

        logging.info(f"Time step: {self.time} Logging BS-Sensor connections...")
        self.log_all_connections_sensors()
     
        # Generate packets for each UE and sensor
        logging.info(f"Time step: {self.time} Package generation starting...")

        for ue in self.users.values():
            self.packet_generation(ue)

        #for sensor in self.sensors.values():
        #    self.packet_generation(sensor)
        
        logging.info(f"Time step: {self.time} Package generation terminated...")

        # Log sensor and ue data uplink queues
        logging.info(f"Time step: {self.time} Device uplink queues...")
        self.log_device_uplink_queue()

        #logging.info(f"Time step: {self.time} Sensor uplink queues...")
        #self.log_sensor_uplink_queue()

        # TODO: add penalties for changing connections?
        #for ue_id, action in actions.items():
        #    self.apply_action(action, self.users[ue_id])

        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)

        # Logging datarates
        logging.info(f"Time step: {self.time} Data rates...")
        self.log_datarates()

        # Packet transmission
        logging.info(f"Time step: {self.time} Packet transmission starting...")
        self.transfer_data() 
        logging.info(f"Time step: {self.time} Packet transmission over...")

        # Log the queue sizes
        self.queue_logs['time'].append(self.time)
        self.queue_logs['sensor_queues'].append({sensor.sensor_id: ue.data_buffer_uplink.current_size for sensor in self.sensors.values()})
        self.queue_logs['ue_queues'].append({ue.ue_id: ue.data_buffer_uplink.current_size for ue in self.users.values()})
        self.queue_logs['bs_queues'].append({bs.bs_id: bs.data_buffer_uplink.current_size for bs in self.stations.values()})

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {
            ue: self.utility.utility(self.macro[ue]) for ue in self.active
        }

        # scale utilities to range [-1, 1] before computing rewards
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)

        # evaluate metrics and update tracked metrics given the core simulation
        self.monitor.update(self)

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
        self.current_time += self.time_step

        # check whether episode is done & close the environment
        if self.time_is_up and self.window:
            self.close()

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.time_is_up:
            return self.step({})

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        # store latest monitored results in `info` dictionary
        info = {**info, **self.monitor.info()}

        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        terminated = False
        truncated = self.time_is_up

        return observation, rewards, terminated, truncated, info

    @property
    def time_is_up(self):
        """Return true after max. time steps or once last UE departed."""
        return self.time >= min(self.EP_MAX_TIME, self.max_departure)

    def macro_datarates(self, datarates):
        """Compute aggregated UE data rates given all its connections."""
        epsilon = 1e-10  # Small value to prevent zero data rates
        ue_datarates = Counter()
        for (bs, ue), datarate in self.datarates.items():
            ue_datarates.update({ue: datarate + epsilon})
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

    def bs_isolines(self, drate: float) -> Dict:
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

        def ue_features(ue: UserEquipment):
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

    def render(self) -> None:
        mode = self.render_mode

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

            ax.scatter(
                ue.point.x,
                ue.point.y,
                s=200,
                zorder=2,
                color=color,
                marker="o",
            )
            ax.annotate(ue.ue_id, xy=(ue.point.x, ue.point.y), ha="center", va="center")

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

        for sensor in self.sensors.values():
            # plot sensor symbol and annonate by its sensor ID
            ax.plot(
                sensor.point.x,
                sensor.point.y,
                marker=SENSOR_SYMBOL,
                markersize=10,
                markeredgewidth=0.1,
                color="blue",
            )
            sensor_id = string.ascii_uppercase[sensor.sensor_id]
            ax.annotate(
                sensor_id,
                xy=(sensor.point.x, sensor.point.y),
                xytext=(0, -15),
                ha="center",
                va="bottom",
                textcoords="offset points",
                fontsize="8",
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
        mean_utilities = self.monitor.scalar_results["mean utility"]
        mean_utility = mean_utilities[-1]
        total_mean_utility = np.mean(mean_utilities)

        mean_datarates = self.monitor.scalar_results["mean datarate"]
        mean_datarate = mean_datarates[-1]
        total_mean_datarate = np.mean(mean_datarates)

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
        mean_utility = self.monitor.scalar_results["mean utility"]
        ax.plot(time, mean_utility, linewidth=1, color="black")

        ax.set_ylabel("Avg. Utility")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([self.utility.lower, self.utility.upper])

    def render_ues_connected(self, ax) -> None:
        time = np.arange(self.time)
        ues_connected = self.monitor.scalar_results["number connected"]
        ax.plot(time, ues_connected, linewidth=1, color="black")

        ax.set_xlabel("Time")
        ax.set_ylabel("#Conn. UEs")
        ax.set_xlim([0.0, self.EP_MAX_TIME])
        ax.set_ylim([0.0, len(self.users)])


    def render_queues(self) -> None:
        if not hasattr(self, 'queue_logs') or not self.queue_logs['time']:
            return

        fig, ax = plt.subplots()
        times = self.queue_logs['time']
        
        # Plot sensor queues
        for sensor_id in self.sensors.keys():
            queue_sizes = [queues.get(sensor_id, 0) for queues in self.queue_logs['sensor_queues']]
            ax.plot(times, queue_sizes, label=f"Sensor {sensor_id}")

        # Plot user device queues
        for ue_id in self.users.keys():
            queue_sizes = [queues.get(ue_id, 0) for queues in self.queue_logs['ue_queues']]
            ax.plot(times, queue_sizes, label=f"UE {ue_id}")

        # Plot base station queues
        for bs_id in self.stations.keys():
            queue_sizes = [queues.get(bs_id, 0) for queues in self.queue_logs['bs_queues']]
            ax.plot(times, queue_sizes, label=f"BS {bs_id}")

        ax.set_title("Queue Sizes Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Queue Size")

        # Move legend to the right side
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


    def close(self) -> None:
        """Closes the environment and terminates its visualization."""
        pygame.quit()
        self.window = None
        self.closed = True

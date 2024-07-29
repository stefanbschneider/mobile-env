import string
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional

import gymnasium
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame import Surface
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
from mobile_env.core.buffers import JobQueue
from mobile_env.core.job_generator import JobGenerator
from mobile_env.core.data_transfer import DataTransferManager
from mobile_env.core.logger import Logger
from mobile_env.handlers.smart_city_handler import MComSmartCityHandler

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
        # store resource allocations for each base station
        self.resource_allocations = None
        # stores each UE's (scaled) utility
        self.utilities: Dict[UserEquipment, float] = None
        # stores each Sensor's (scaled) utility
        self.utilities_sensor: Dict[Sensor, float] = None
        # define RNG (as of now: unused)
        self.rng = None

        # Instantiate JobGenerator and DataTransferManager
        self.job_generator = JobGenerator(self)
        self.data_transfer_manager = DataTransferManager(self)

        # Instantiate the Logger class
        self.logger = Logger(self)

        # Initialize or update queue logs
        if not hasattr(self, 'queue_logs'):
            self.queue_logs = {
                'time': [],
                'bs_uplink_ue_queues': [],
                'bs_downlink_ue_queues': [],
                'bs_uplink_sensor_queues': [],
                'bs_downlink_sensor_queues': [],
                'ue_uplink_queues': [],
                'sensor_uplink_queues': []
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
            "handler": MComSmartCityHandler,
            # default cell config
            "bs": {
                "bw": 20e6, 
                "freq": 2500, 
                "tx": 40, 
                "height": 50, 
                "computational_power": 200,
            },
            # default UE config
            "ue": {
                "velocity": 1.5,
                "snr_tr": 2e-8,
                "noise": 1e-9,
                "height": 1.5,
            },
            # default Sensor config
            "sensor": {
                "height": 1.5,
                "snr_tr": 2e-8,
                "noise": 1e-9,
                "velocity": 0,
                "radius": 500,
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

    def apply_action(self, bs: BaseStation, bandwidth_allocation: float, computational_allocation: float) -> Tuple[float, float, float, float]:
        """Allocate bandwidth and computational resource between UEs and sensors."""
        if not (0 <= bandwidth_allocation <= 1 and 0 <= computational_allocation <= 1):
            raise ValueError("Allocations must be between 0 and 1")
        
        # Apply bandwidth resource allocation to UEs and sensors
        ue_bandwidth = bs.bw * bandwidth_allocation
        sensor_bandwidth = bs.bw * (1 - bandwidth_allocation)

        # Apply computational resource allocation to UEs and sensors
        ue_computational_power = bs.computational_power * computational_allocation
        sensor_computational_power = bs.computational_power * (1 - computational_allocation)

        logging.info(f"Time step: {self.time} Bandwidth allocated to UEs: {ue_bandwidth} Hz, to Sensors: {sensor_bandwidth} Hz")
        logging.info(f"Time step: {self.time} Computational power allocated to UEs: {ue_computational_power} FLOPS, to Sensors: {sensor_computational_power} FLOPS")

        return ue_bandwidth, sensor_bandwidth, ue_computational_power, sensor_computational_power

    def check_connectivity(self, bs: BaseStation, ue: UserEquipment) -> bool:
        """Connection can be established if SNR exceeds threshold of UE."""
        snr = self.channel.snr(bs, ue)
        return snr > ue.snr_threshold
    
    def check_connectivity_sensor(self, bs: BaseStation, sensor: Sensor) -> bool:
        """Connection can be established if SNR exceeds threshold of sensor."""
        snr = self.channel.snr(bs, sensor)
        return snr > sensor.snr_threshold

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

    def update_connections_sensors(self) -> None:
        """Release connections where BS and sensor is out-of-range."""
        connections_sensor = {
            bs: set(sensor for sensor in sensors if self.check_connectivity_sensor(bs, sensor))
            for bs, sensors in self.connections_sensor.items()
        }
        self.connections_sensor.clear()
        self.connections_sensor.update(connections_sensor)

    def connect_bs_ue(self) -> None:
        """Connect the UE to the closest base station within data range."""
        for ue in self.users.values():
            closest_bs: Optional[BaseStation] = None    # Initialize the closest base station to None
            min_distance = float('inf')                 # Initialize the minimum distance to infinity

            # Iterate through all base stations to find the closest one
            for bs in self.stations.values():
                # Calculate the Euclidean distance between the UE and the base station
                distance = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2)
                # Update the closest base station if the current one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs

            # If a closest base station is found, establish the connection
            if closest_bs:
                if closest_bs not in self.connections:
                    self.connections[closest_bs] = set()
                self.connections[closest_bs].add(ue)

    def connect_bs_sensor(self) -> None:
        """Connect each sensor to the closest base station."""
        for sensor in self.sensors.values():
            closest_bs: Optional[BaseStation] = None     # Initialize the closest base station to None
            min_distance = float('inf')                  # Initialize the minimum distance to infinity
            
            #  Iterate through all base stations to find the closest one
            for bs in self.stations.values():
                # Calculate the Euclidean distance between the sensor and the base station
                distance = np.sqrt((sensor.x - bs.x) ** 2 + (sensor.y - bs.y) ** 2)
                # Update the closest base station if the current one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs

            # If a closest base station is found, establish the connection
            if closest_bs:
                sensor.connected_base_station = closest_bs
                if closest_bs not in self.connections_sensor:
                    self.connections_sensor[closest_bs] = set()
                self.connections_sensor[closest_bs].add(sensor)

    def update_sensor_logs(self):
        '''Checks if UE is in range and adds the timestamp to the log of the UE'''
        for ue in self.users.values(): 
            ue_point = ue.point

            for sensor_id, sensor in self.sensors.items():
                if sensor.is_within_range(ue_point):
                    # Ensure ue.ue_id is a string if sensor.logs uses string keys
                    ue_id_str = str(ue.ue_id)
                    # Check if the UE is detected, initialize a list if not
                    if ue_id_str not in sensor.logs:
                        sensor.logs[ue_id_str] = [self.time]
                    else:
                        # If the UE has already been detected, append the current time
                        sensor.logs[ue_id_str].append(self.time)
            
            # Log the current state of all sensor logs after processing each UE
            #logging.info(f"Sensor {sensor_id} logs after processing UE {ue.ue_id}: {sensor.logs}")

    def step(self, actions: Tuple[float, float]):
        assert not self.time_is_up, "step() called on terminated episode"

        logging.info(f"Time step: {self.time} Establishing connections...")

        # connect UEs and sensors to the closest base station and establish connection
        self.connect_bs_ue()
        self.connect_bs_sensor()

        # check snr thresholds, release established connections that moved e.g. out-of-range
        self.update_connections()
        self.update_connections_sensors()

        # update UE positions in sensor logs
        self.update_sensor_logs()

        # Logging base station connections
        self.logger.log_all_connections()
     
        # Generate jobs for each UE and sensor
        logging.info(f"Time step: {self.time} Job generation starting...")

        for ue in self.users.values():
            self.job_generator.generate_job_ue(ue)
        for sensor in self.sensors.values():
            self.job_generator.generate_job_sensor(sensor)

        logging.info(f"Time step: {self.time} Job generation terminated...")

        # Log sensor and ue data queues
        self.logger.log_all_queues()

        # apply handler to transform actions to expected shape
        bandwidth_allocation, computational_allocation = self.handler.action(self, actions)
        logging.info(f"Time step: {self.time} Communication allocation to UEs in percentage: {bandwidth_allocation}")
        logging.info(f"Time step: {self.time} Computation allocation to UEs in percentage: {computational_allocation}")

        self.resource_allocations = {}
        for bs in self.stations.values():
            bandwidth_for_ues, bandwidth_for_sensors, computational_power_for_ues, computational_power_for_sensors = self.apply_action(bs, bandwidth_allocation, computational_allocation)
            # Store the resource allocations for each BS in the dictionary
            
            self.resource_allocations[bs] = {
                'bandwidth_for_ues': bandwidth_for_ues,
                'bandwidth_for_sensors': bandwidth_for_sensors,
                'computational_power_for_ues': computational_power_for_ues,
                'computational_power_for_sensors': computational_power_for_sensors
            }
            
        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations.values():
            bs_bandwidth = self.resource_allocations[bs]['bandwidth_for_ues']
            drates_ue = self.station_allocation(bs, bs_bandwidth)
            self.datarates.update(drates_ue)

        self.datarates_sensor = {}
        for bs in self.stations.values():
            bs_bandwidth = self.resource_allocations[bs]['bandwidth_for_sensors']
            drates_sensor = self.station_allocation_sensor(bs, bs_bandwidth)
            self.datarates_sensor.update(drates_sensor)
        
        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)

        # Logging datarates
        self.logger.log_all_datarates()

        # Packet uplink transmission
        logging.info(f"Time step: {self.time} Job transfer uplink starting...")
        self.data_transfer_manager.transfer_data_uplink()
        logging.info(f"Time step: {self.time} Job transfer uplink over...")

        # Log sensor and ue data queues
        self.logger.log_all_queues()

        # Process data in MEC servers
        logging.info(f"Time step: {self.time} Data processing starting...")
        self.data_transfer_manager.process_data_mec(computational_power_for_ues, computational_power_for_sensors)
        logging.info(f"Time step: {self.time} Data processing over...")

        # Log sensor and ue data queues
        self.logger.log_all_queues()

        # Log queue sizes
        #self.log_queue_sizes()

        # Log the job data frame
        self.job_generator.log_packets_ue()
        self.job_generator.log_packets_sensor()

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
        # TODO: implement a new way to compute rewards because this strategy computes rewards according to utility for each EU based on the datarate
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

    def station_allocation(self, bs: BaseStation, bandwidth_for_ues: float) -> Dict:
        """Schedule BS's resources (e.g. phy. res. blocks) to connected UEs."""
        conns = self.connections[bs]

        # compute SNR & max. data rate for each connected user equipment
        snrs = [self.channel.snr(bs, ue) for ue in conns]

        # UE's max. data rate achievable when BS schedules all resources to it
        max_allocation = [
            self.channel.datarate_split(bs, ue, snr, bandwidth_for_ues) for snr, ue in zip(snrs, conns)
        ]
        # BS shares resources among connected user equipments
        rates = self.scheduler.share_ue(bs, max_allocation, bandwidth_for_ues)

        return {(bs, ue): rate for ue, rate in zip(conns, rates)}
    
    def station_allocation_sensor(self, bs: BaseStation, bandwidth_for_sensors: float) -> Dict:
        """Schedule BS's resources (e.g. phy. res. blocks) to connected sensors."""
        conns_sensor = self.connections_sensor[bs]

        # compute SNR & max. data rate for each connected sensor
        snrs_sensor = [self.channel.snr(bs, sensor) for sensor in conns_sensor]

        # Sensor's max. data rate achievable when BS schedules all resources to it
        max_allocation = [
            self.channel.datarate_split(bs, sensor, snr, bandwidth_for_sensors) for snr, sensor in zip(snrs_sensor, conns_sensor)
        ]

        # BS shares resources among connected sensors
        rates_sensor = self.scheduler.share_sensor(bs, max_allocation, bandwidth_for_sensors)

        return {(bs, sensor): rate for sensor, rate in zip(conns_sensor, rates_sensor)}

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

        # calculate isoline contours for BSs' connectivity range (1 MB/s range)
        if self.conn_isolines is None:
            self.conn_isolines = self.bs_isolines(1.0)
        # calculate isoline contours for BSs' 50 MB/s range
        if self.mb_isolines is None:
            self.mb_isolines = self.bs_isolines(100.0)

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
        ax.set_ylim([-1, 1])

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

    def log_queue_sizes(self):
        """Logs the current queue sizes of all entities."""
        self.queue_logs['time'].append(self.time)

        bs_uplink_ue_sizes = [bs.data_buffer_uplink_ue.data_queue.qsize() for bs in self.stations.values()]
        self.queue_logs['bs_uplink_ue_queues'].append(bs_uplink_ue_sizes)

        bs_downlink_ue_sizes = [bs.data_buffer_downlink_ue.data_queue.qsize() for bs in self.stations.values()]
        self.queue_logs['bs_downlink_ue_queues'].append(bs_downlink_ue_sizes)

        bs_uplink_sensor_sizes = [bs.data_buffer_uplink_sensor.data_queue.qsize() for bs in self.stations.values()]
        self.queue_logs['bs_uplink_sensor_queues'].append(bs_uplink_sensor_sizes)

        bs_downlink_sensor_sizes = [bs.data_buffer_downlink_sensor.data_queue.qsize() for bs in self.stations.values()]
        self.queue_logs['bs_downlink_sensor_queues'].append(bs_downlink_sensor_sizes)

        ue_uplink_sizes = [ue.data_buffer_uplink.data_queue.qsize() for ue in self.users.values()]
        self.queue_logs['ue_uplink_queues'].append(ue_uplink_sizes)

        sensor_uplink_sizes = [sensor.data_buffer_uplink.data_queue.qsize() for sensor in self.sensors.values()]
        self.queue_logs['sensor_uplink_queues'].append(sensor_uplink_sizes)

    def close(self) -> None:
        """Closes the environment and terminates its visualization."""
        pygame.quit()
        self.window = None
        self.closed = True

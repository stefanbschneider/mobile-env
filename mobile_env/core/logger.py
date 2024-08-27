import logging

class Logger:
    def __init__(self, env):
        self.env = env
        
        # Set up the first logger for general simulation logs
        self.simulation_logger = logging.getLogger('simulation_logger')
        self.simulation_logger.setLevel(logging.INFO)
        sim_handler = logging.FileHandler('simulation.log')
        sim_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        sim_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.simulation_logger.addHandler(sim_handler)

        # Set up the second logger for performance logs
        self.reward_logger = logging.getLogger('reward_logger')
        self.reward_logger.setLevel(logging.INFO)
        reward_handler = logging.FileHandler('rewards.log')
        reward_handler.setLevel(logging.INFO)
        #perf_handler.setFormatter(formatter)
        self.reward_logger.addHandler(reward_handler)

    def log_simulation(self, message: str) -> None:
        """Logs simulation steps."""
        self.simulation_logger.info(message)

    def log_reward(self, message: str) -> None:
        """Logs reward computation steps."""
        self.reward_logger.info(message)

    def log_datarates(self) -> None:
        """Logs data transfer rates of each connected ue-bs pair."""
        for (bs, ue), rate in sorted(self.env.datarates.items(), key=lambda x: x[0][1].ue_id):
            self.log_simulation(f"Time step: {self.env.time} Data transfer rate for {ue} connected to {bs} is : {rate} Mbps")

    def log_datarates_sensor(self) -> None:
        """Logs data transfer rates of each connected sensor-bs pair."""
        for (bs, sensor), rate in sorted(self.env.datarates_sensor.items(), key=lambda x: x[0][1].sensor_id):
            self.log_simulation(f"Time step: {self.env.time} Data transfer rate for {sensor} connected to {bs} is : {rate} Mbps")

    def log_device_uplink_queue(self) -> None:
        """Logs the job indexes, initial sizes, and remaining sizes for every job in the uplink buffer of UEs."""
        for ue in self.env.users.values():
            buffer_size = ue.data_buffer_uplink.data_queue.qsize()
            if buffer_size > 0:
                for job in list(ue.data_buffer_uplink.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} Device: {ue.ue_id}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB, Remaining size: {job['remaining_request_size']} MB, "
                        f"Computation request: {job['computation_request']} FLOPS"
                    )

    def log_sensor_uplink_queue(self) -> None:
        """Logs the job indexes, initial sizes, and remaining sizes for every job in the uplink buffer of sensors."""
        for sensor in self.env.sensors.values():
            buffer_size = sensor.data_buffer_uplink.data_queue.qsize()
            if buffer_size > 0:
                for job in list(sensor.data_buffer_uplink.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} Sensor: {sensor.sensor_id}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB, Remaining size: {job['remaining_request_size']} MB, "
                        f"Computation request: {job['computation_request']} FLOPS"
                    )                 
                    
    def log_bs_transferred_jobs_queue(self) -> None:
        """Logs the job indexes, initial sizes, and remaining sizes for every job in the BS uplink queues."""
        for bs in self.env.stations.values():
            # Log jobs from user devices
            buffer_size_ue = bs.transferred_jobs_ue.data_queue.qsize()
            if buffer_size_ue > 0:
                for job in list(bs.transferred_jobs_ue.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} BS: {bs.bs_id}, UE: {job['device_id']}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB and Computational request:{job['computation_request']} FLOPS"
                    )

            # Log jobs from sensors
            buffer_size_sensor = bs.transferred_jobs_sensor.data_queue.qsize()
            if buffer_size_sensor > 0:
                for job in list(bs.transferred_jobs_sensor.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} BS: {bs.bs_id}, Sensor: {job['device_id']}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB and Computational request:{job['computation_request']} FLOPS"
                    )

    def log_bs_accomplished_jobs_queue(self) -> None:
        """Logs the job indexes, initial sizes, and remaining sizes for every job in the BS downlink queues."""
        for bs in self.env.stations.values():
            # Log jobs from user devices
            buffer_size_ue = bs.accomplished_jobs_ue.data_queue.qsize()
            if buffer_size_ue > 0:
                for job in list(bs.accomplished_jobs_ue.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} BS: {bs.bs_id}, UE: {job['device_id']}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB and Computational request:{job['computation_request']} FLOPS"
                    )

            # Log jobs from sensors
            buffer_size_sensor = bs.accomplished_jobs_sensor.data_queue.qsize()
            if buffer_size_sensor > 0:
                for job in list(bs.accomplished_jobs_sensor.data_queue.queue):
                    self.log_simulation(
                        f"Time step: {self.env.time} BS: {bs.bs_id}, Sensor: {job['device_id']}, Job index: {job['packet_id']}, "
                        f"Initial size: {job['initial_request_size']} MB and Computational request:{job['computation_request']} FLOPS"
                    )

    def log_connections(self) -> None:
        """Logs connections between base stations and user equipment."""
        connection_strings = [
            f"BS {bs.bs_id}: [{','.join(map(str, sorted(ue.ue_id for ue in ues)))}]"
            for bs, ues in self.env.connections.items()
        ]
        log_message = "Connections UEs: " + "; ".join(connection_strings)
        self.log_simulation(f"Time step: {self.env.time} {log_message}")

    def log_connections_sensors(self) -> None:
        """Logs connections between base stations and sensors."""
        connection_strings = [
            f"BS {bs.bs_id}: [{','.join(map(str, sorted(sensor.sensor_id for sensor in sensors)))}]"
            for bs, sensors in self.env.connections_sensor.items()
        ]
        log_message = "Connections Sensors: " + "; ".join(connection_strings)
        self.log_simulation(f"Time step: {self.env.time} {log_message}")

    def log_all_connections(self) -> None:
        """Log all connections between base stations and user equipment, as well as sensors."""
        self.log_simulation(f"Time step: {self.env.time} Logging BS-UE connections...")
        self.log_connections()
        self.log_simulation(f"Time step: {self.env.time} Logging BS-Sensor connections...")
        self.log_connections_sensors()

    def log_all_queues(self) -> None:
        """Log all job queues across devices, sensors, and base stations."""
        self.log_simulation(f"Time step: {self.env.time} Device uplink queues...")
        self.log_device_uplink_queue()
        self.log_simulation(f"Time step: {self.env.time} Sensor uplink queues...")
        self.log_sensor_uplink_queue()
        self.log_simulation(f"Time step: {self.env.time} Base station queue for transferred jobs...")
        self.log_bs_transferred_jobs_queue()
        self.log_simulation(f"Time step: {self.env.time} Base station queue for accomplished jobs...")
        self.log_bs_accomplished_jobs_queue()

    def log_all_datarates(self) -> None:
        """Log data transfer rates for all connected pairs of UEs and sensors."""
        self.log_simulation(f"Time step: {self.env.time} Data rates...")
        self.log_datarates()
        self.log_datarates_sensor()
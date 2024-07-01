import logging

class Logger:
    def __init__(self, env):
        self.env = env

    def log_datarates(self) -> None:
        # Logs data transfer rates of each connected ue-bs pair.
        for (bs, ue), rate in sorted(self.env.datarates.items(), key=lambda x: x[0][1].ue_id):
            logging.info(f"Time step: {self.env.time} Data transfer rate for {ue} connected to {bs} is : {rate}")

    def log_datarates_sensor(self) -> None:
        # Logs data transfer rates of each connected sensor-bs pair.
        for (bs, sensor), rate in sorted(self.env.datarates_sensor.items(), key=lambda x: x[0][1].sensor_id):
            logging.info(f"Time step: {self.env.time} Data transfer rate for {sensor} connected to {bs} is : {rate}")

    def log_device_uplink_queue(self) -> None:
        # Logs the job indexes, initial sizes, and remaining sizes for every job in the uplink buffer of UEs.
        for ue in self.env.users.values():
            buffer_size = ue.data_buffer_uplink.data_queue.qsize()
            if buffer_size > 0:
                for job in list(ue.data_buffer_uplink.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Device: {ue.ue_id}, Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                        )

    def log_sensor_uplink_queue(self) -> None:
        # Logs the job indexes, initial sizes, and remaining sizes for every job in the uplink buffer of sensors.
        for sensor in self.env.sensors.values():
            buffer_size = sensor.data_buffer_uplink.data_queue.qsize()
            if buffer_size > 0:
                for job in list(sensor.data_buffer_uplink.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Sensor: {sensor.sensor_id}, Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                        )                 
                    
    def log_bs_uplink_queue(self) -> None:
        # Logs the job indexes, initial sizes, and remaining sizes for every job in the BS queues.
        for bs in self.env.stations.values():
            # Log jobs from user devices
            buffer_size_ue = bs.data_buffer_uplink_ue.data_queue.qsize()
            if buffer_size_ue > 0:
                for job in list(bs.data_buffer_uplink_ue.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Base station: {bs.bs_id}, Device Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                    )

            # Log jobs from sensors
            buffer_size_sensor = bs.data_buffer_uplink_sensor.data_queue.qsize()
            if buffer_size_sensor > 0:
                for job in list(bs.data_buffer_uplink_sensor.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Base station: {bs.bs_id}, Sensor Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                    )

    def log_bs_downlink_queue(self) -> None:
        # Logs the job indexes, initial sizes, and remaining sizes for every job in the BS queues.
        for bs in self.env.stations.values():
            # Log jobs from user devices
            buffer_size_ue = bs.data_buffer_downlink_ue.data_queue.qsize()
            if buffer_size_ue > 0:
                for job in list(bs.data_buffer_downlink_ue.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Base station: {bs.bs_id}, Device Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                    )

            # Log jobs from sensors
            buffer_size_sensor = bs.data_buffer_downlink_sensor.data_queue.qsize()
            if buffer_size_sensor > 0:
                for job in list(bs.data_buffer_downlink_sensor.data_queue.queue):
                    logging.info(
                        f"Time step: {self.env.time} Base station: {bs.bs_id}, Sensor Job index: {job['index']}, "
                        f"Initial size: {job['initial_size']}, Remaining size: {job['remaining_size']}"
                    )

    def log_all_connections(self) -> None:
        # Log all connections between base stations and user equipment in one line.
        connection_strings = [
            f"BS {bs.bs_id}: [{','.join(map(str, sorted(ue.ue_id for ue in ues)))}]"
            for bs, ues in self.env.connections.items()
        ]
        log_message = "Connections UEs: " + "; ".join(connection_strings)
        logging.info(log_message)

    def log_all_connections_sensors(self) -> None:
        # Log all connections between base stations and sensors in one line.
        connection_strings = [
            f"BS {bs.bs_id}: [{','.join(map(str, sorted(sensor.sensor_id for sensor in sensors)))}]"
            for bs, sensors in self.env.connections_sensor.items()
        ]
        log_message = "Connections Sensors: " + "; ".join(connection_strings)
        logging.info(log_message)

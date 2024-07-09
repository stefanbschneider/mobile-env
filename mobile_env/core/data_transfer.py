import logging
from typing import Dict, Union, List, Tuple
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.buffers import Buffer

class DataTransferManager:
    def __init__(self, env):
        self.env = env

    def transfer_uplink_data(self) -> None:
        # Transfers data from UEs and sensors to base stations according to data rates.
        for bs in self.env.stations.values():
            self._transfer_data_to_bs(bs, self.env.connections.get(bs, []), 'uplink')  # Improved method naming
            self._transfer_data_to_bs(bs, self.env.connections_sensor.get(bs, []), 'uplink')  # Improved method naming

    def process_data_mec(self, ue_computational_power: float, sensor_computational_power: float) -> None:
        # Process data in MEC servers and write processed jobs into downlink queues.
        for bs in self.env.stations.values():
            self._process_data_for_bs(bs, ue_computational_power, sensor_computational_power)

    def _transfer_data_to_bs(self, bs: BaseStation, devices: List[Union[UserEquipment, Sensor]], direction: str) -> None:
        for device in devices:
            self._transfer_data(device, bs, direction)

    def _transfer_data(self, src: Union[UserEquipment, BaseStation], dst: Union[BaseStation, UserEquipment, Sensor], direction: str) -> None:
        remaining_data_transfer_rate = self._get_remaining_data_rate(src, dst, direction)
        src_buffer, dst_buffer = self._get_buffers(src, dst, direction)

        if remaining_data_transfer_rate <= 0:
            logging.warning(f"No data rate for {direction} connection from {src} to {dst}. Packet transmission aborted.")
            return

        while remaining_data_transfer_rate > 0 and not src_buffer.data_queue.empty():
            job = src_buffer.get_job()

            if job['serving_time_start'] is None:
                job['serving_time_start'] = self.env.time

            bits_to_send = min(job['remaining_size'], remaining_data_transfer_rate)
            self._update_job(job, bits_to_send)

            remaining_data_transfer_rate -= bits_to_send

            logging.info(
                f"Time step: {self.env.time} From: {src} To: {dst}, Job index: {job['index']} "
                f"Bits sent: {bits_to_send}, Remaining size: {job['remaining_size']}"
            )

            if job['remaining_size'] <= 0:
                src_buffer.remove_job()
                dst_buffer.add_job(job)
                job['serving_time_end'] = self.env.time
                job['serving_time_total'] = job['serving_time_end'] - job['serving_time_start']
                logging.info(
                    f"Time step: {self.env.time} Job {job['index']} transferred from {src} to {dst} with serving time {job['serving_time_total']}."
                )
            else:
                logging.info(
                    f"Time step: {self.env.time} Job {job['index']} partially transferred from {src} to {dst}."
                )
                break

    def _find_device_by_id(self, device_id: int, device_type: str) -> Union[UserEquipment, Sensor, None]:
        # Find device by ID and type.
        if device_type == 'user_device':
            return self.env.users.get(device_id)
        elif device_type == 'sensor':
            return self.env.sensors.get(device_id)
        return None

    def _get_remaining_data_rate(self, src: Union[UserEquipment, Sensor], dst: BaseStation, direction: str) -> float:
        if direction == 'uplink':
            if isinstance(src, UserEquipment):
                return self.env.datarates.get((dst, src), 1e6)
            elif isinstance(src, Sensor):
                return self.env.datarates_sensor.get((dst, src), 1e6)
        raise ValueError(f"Invalid direction: {direction}")

    def _get_buffers(self, src: Union[BaseStation, UserEquipment, Sensor], dst: Union[BaseStation, UserEquipment, Sensor], direction: str) -> Tuple[Buffer, Buffer]:
        if direction == 'uplink':
            src_buffer = src.data_buffer_uplink
            dst_buffer = dst.data_buffer_uplink_ue if isinstance(src, UserEquipment) else dst.data_buffer_uplink_sensor
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return src_buffer, dst_buffer

    def _process_data_for_bs(self, bs: BaseStation, ue_computational_power: float, sensor_computational_power: float) -> None:
        self._process_data(bs.data_buffer_uplink_ue, bs.data_buffer_downlink_ue, ue_computational_power)
        logging.warn(f"Time step: {self.env.time} UE jobs are processed.")
        self._process_data(bs.data_buffer_uplink_sensor, bs.data_buffer_downlink_sensor, sensor_computational_power)
        logging.warn(f"Time step: {self.env.time} Sensor jobs are processed.")

    def _process_data(self, uplink_buffer: Buffer, downlink_buffer: Buffer, computational_power: float) -> None:
        # Process jobs based on the computational power available at the base station.
        if computational_power <= 0:
            logging.warning(f"No computational power available at the MEC server of the base station.")
            return

        while not uplink_buffer.data_queue.empty() and computational_power > 0:
            job = uplink_buffer.get_job()  # Peek at the job without removing it
            if job and job['computational_requirement'] <= computational_power:
                uplink_buffer.remove_job()  # Now remove the job
                processing_time = job['computational_requirement'] / computational_power  # seconds

                job['processing_time'] = processing_time
                computational_power -= job['computational_requirement']

                # Add the job to the downlink buffer
                downlink_buffer.add_job(job)

                logging.info(
                    f"Time step: {self.env.time} Processed job {job['index']} with computational requirement {job['computational_requirement']}."
                )

                if computational_power < 0:
                    logging.warning("MEC server computational power exhausted. Some jobs may be delayed.")
                    break  # Exit the loop if computational power is exhausted
            else:
                logging.warning(f"Job {job['index']} requires more computational power than available. Skipping job.")
                break

    def _update_job(self, job: Dict[str, Union[int, float]], bits_to_send: float) -> None:
        job['remaining_size'] -= bits_to_send
        if job['remaining_size'] < 0:
            job['remaining_size'] = 0

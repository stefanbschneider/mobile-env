import logging
from typing import Dict, Union, List, Tuple
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.buffers import JobQueue

class DataTransferManager:
    def __init__(self, env):
        self.env = env

    def transfer_data_uplink(self) -> None:
        # Transfers data from UEs and sensors to base stations according to data rates.
        for bs in self.env.stations.values():
            self._transfer_data_to_bs(bs, self.env.connections.get(bs, []), 'uplink')
            self._transfer_data_to_bs(bs, self.env.connections_sensor.get(bs, []), 'uplink')

    def process_data_mec(self, computational_power_for_ues: Dict[int, float], computational_power_for_sensors: Dict[int, float]) -> None:
        # Process data in MEC servers and write processed jobs into downlink queues.
        for bs in self.env.stations.values():
            self._process_data_for_bs(bs)

    def _transfer_data_to_bs(self, bs: BaseStation, devices: List[Union[UserEquipment, Sensor]], direction: str) -> None:
        for device in devices:
            self._transfer_data(device, bs, direction)

    def _transfer_data(self, src, dst, direction):
        if direction == 'uplink':
            src_buffer = src.data_buffer_uplink
            dst_buffer = dst.data_buffer_uplink_ue if isinstance(src, UserEquipment) else dst.data_buffer_uplink_sensor
            data_rate = self.env.datarates.get((src, dst), 0)
        elif direction == 'downlink':
            src_buffer = src.data_buffer_downlink_ue if isinstance(dst, UserEquipment) else src.data_buffer_downlink_sensor
            dst_buffer = dst.data_buffer_downlink
            data_rate = self.env.datarates.get((src, dst), 0)
        else:
            raise ValueError("Invalid direction. Expected 'uplink' or 'downlink'.")

        # Convert data rate from bits per second to bits per time step
        remaining_data_transfer_rate = data_rate * self.env.time_step

        while remaining_data_transfer_rate > 0 and not src_buffer.is_empty():
            job = src_buffer.peek_job()

            # Set serving_time_start if not already set
            if job['serving_time_start'] is None:
                job['serving_time_start'] = self.env.time

            bits_to_send = min(job['remaining_size'], remaining_data_transfer_rate)

            self._update_job(job, bits_to_send)

            remaining_data_transfer_rate -= bits_to_send

            # If the job is completely transferred
            if job['remaining_size'] <= 0:
                src_buffer.dequeue_job()
                dst_buffer.enqueue_job(job)
                job['serving_time_end'] = self.env.time
                job['serving_time_total'] = job['serving_time_end'] - job['serving_time_start']

                logging.info(
                    f"Time step: {self.env.time} Job {job['packet_id']} transferred from {src} to {dst} with serving time {job['serving_time_total']}."
                )
            else:
                logging.info(
                    f"Time step: {self.env.time} Job {job['packet_id']} partially transferred from {src} to {dst}."
                )



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

    def _get_buffers(self, src: Union[BaseStation, UserEquipment, Sensor], dst: Union[BaseStation, UserEquipment, Sensor], direction: str) -> Tuple[JobQueue, JobQueue]:
        if direction == 'uplink':
            src_buffer = src.data_buffer_uplink
            if isinstance(src, UserEquipment):
                dst_buffer = dst.data_buffer_uplink_ue
            elif isinstance(src, Sensor):
                dst_buffer = dst.data_buffer_uplink_sensor
            else:
                raise ValueError(f"Invalid source type: {type(src)}")
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return src_buffer, dst_buffer

    def _process_data_for_bs(self, bs: BaseStation) -> None:
        self._process_data(bs.data_buffer_uplink_ue, bs.data_buffer_downlink_ue, bs.computational_power)
        logging.warning(f"Time step: {self.env.time} UE jobs are processed.")
        self._process_data(bs.data_buffer_uplink_sensor, bs.data_buffer_downlink_sensor, bs.computational_power)
        logging.warning(f"Time step: {self.env.time} Sensor jobs are processed.")

    def _process_data(self, uplink_buffer, downlink_buffer, computational_power):
        if computational_power <= 0:
            return

        while not uplink_buffer.is_empty() and computational_power > 0:
            job = uplink_buffer.peek_job()  # Peek at the job without removing it

            if job is None:
                break  # No job to process

            computation_needed = job['computation_request']

            if computation_needed <= computational_power:
                uplink_buffer.dequeue_job()  # Remove the job from the queue
                processing_time = computation_needed / computational_power  # Compute processing time

                job['processing_time'] = processing_time
                computational_power -= computation_needed

                # Add the job to the downlink buffer
                downlink_buffer.enqueue_job(job)

                logging.info(
                    f"Time step: {self.env.time} Processed job {job['packet_id']} with computational requirement {computation_needed}."
                )

                if computational_power <= 0:
                    logging.warning("MEC server computational power exhausted. Some jobs may be delayed.")
                    break  # Exit the loop if computational power is exhausted
            else:
                logging.warning(f"Job {job['packet_id']} requires more computational power than available. Skipping job.")
                break


    def _update_job(self, job: Dict[str, Union[int, float]], bits_to_send: float) -> None:
        job['remaining_size'] -= bits_to_send
        if job['remaining_size'] < 0:
            job['remaining_size'] = 0

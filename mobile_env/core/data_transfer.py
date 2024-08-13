import logging
from typing import Dict, Union, List, Tuple
from mobile_env.core.entities import BaseStation, UserEquipment, Sensor
from mobile_env.core.buffers import JobQueue

Device = Union[UserEquipment, Sensor]

class DataTransferManager:
    def __init__(self, env):
        self.env = env
        self.job_generator = env.job_generator

    def transfer_data_uplink(self) -> None:
        # Transfers data from UEs and sensors to base stations according to data rates.
        for bs in self.env.stations.values():
            self._transfer_data_to_bs(bs, self.env.connections.get(bs, []))
            self._transfer_data_to_bs(bs, self.env.connections_sensor.get(bs, []))

    def process_data_mec(self, ue_computational_power: float, sensor_computational_power: float) -> None:
        # Process data in MEC servers and write processed jobs into downlink queues.
        for bs in self.env.stations.values():
            self._process_data_for_bs(bs, ue_computational_power, sensor_computational_power)

    def _transfer_data_to_bs(self, bs: BaseStation, devices: List[Device]) -> None:
        for device in devices:
            self._transfer_data(device, bs)

    def _transfer_data(self, src: Device, dst: BaseStation) -> None:
        data_transfer_rate = self._get_remaining_data_rate(src, dst)
        src_buffer, dst_buffer = self._get_buffers(src, dst)

        if data_transfer_rate <= 0:
            logging.warning(
                f"Time step: {self.env.time} No data rate for uplink connection from {src} to {dst} "
                f"for device {Device} and bs {BaseStation}. Packet transmission aborted."
            )
            return

        while data_transfer_rate > 0 and not src_buffer.data_queue.empty():
            job = src_buffer.peek_job()

            if job['serving_time_start'] is None:
                job['serving_time_start'] = self.env.time

            bits_to_send = min(job['remaining_request_size'], data_transfer_rate)
            self._update_job(job, bits_to_send)

            data_transfer_rate -= bits_to_send

            logging.info(
                f"Time step: {self.env.time} from {src} to {dst}, job index: {job['packet_id']}, "
                f"Bits sent: {bits_to_send}, Remaining size: {job['remaining_request_size']}"
            )

            if job['remaining_request_size'] <= 0:
                src_buffer.dequeue_job()
                dst_buffer.enqueue_job(job)
                job['serving_time_end'] = self.env.time
                job['serving_time_total'] = job['serving_time_end'] - job['serving_time_start']

                # Update arrival_time of the job in the data frame
                if job['device_type'] == 'user_device':
                    self.job_generator.packet_df_ue.loc[self.job_generator.packet_df_ue['packet_id'] == job['packet_id'], 'arrival_time'] = self.env.time
                elif job['device_type'] == 'sensor':
                    self.job_generator.packet_df_sensor.loc[self.job_generator.packet_df_sensor['packet_id'] == job['packet_id'], 'arrival_time'] = self.env.time
                else:
                    logging.warning(f"Unknown device type {job['device_type']}. Arrival time not updated.")

                logging.info(
                    f"Time step: {self.env.time} Job {job['packet_id']} transferred from {src} to {dst} with serving time {job['serving_time_total']}."
                )
            else:
                logging.info(
                    f"Time step: {self.env.time} Job {job['packet_id']} partially transferred from {src} to {dst}."
                )
                break

    def _get_remaining_data_rate(self, src: Union[UserEquipment, Sensor], dst: BaseStation) -> float:
        if isinstance(src, UserEquipment):
            return self.env.datarates.get((dst, src), 1e6)
        elif isinstance(src, Sensor):
            return self.env.datarates_sensor.get((dst, src), 1e6)
        else:
            raise ValueError(f"Invalid Device")

    def _get_buffers(self, src: Union[UserEquipment, Sensor], dst: BaseStation) -> Tuple[JobQueue, JobQueue]:
        src_buffer = src.data_buffer_uplink
        dst_buffer = dst.transferred_jobs_ue if isinstance(src, UserEquipment) else dst.transferred_jobs_sensor

        return src_buffer, dst_buffer

    def _process_data_for_bs(self, bs: BaseStation, ue_computational_power: float, sensor_computational_power: float) -> None:
        self._process_data(bs.transferred_jobs_ue, bs.accomplished_jobs_ue, ue_computational_power)
        logging.warn(f"Time step: {self.env.time} UE jobs are processed.")
        self._process_data(bs.transferred_jobs_sensor, bs.accomplished_jobs_sensor, sensor_computational_power)
        logging.warn(f"Time step: {self.env.time} Sensor jobs are processed.")

    def _process_data(self, transferred_jobs_queue: JobQueue, accomplished_jobs_queue: JobQueue, computational_power: float) -> None:
        # Process jobs based on the computational power available at the base station.
        if computational_power <= 0:
            logging.warning(f"No computational power available at the MEC server of the base station.")
            return

        while not transferred_jobs_queue.data_queue.empty() and computational_power > 0:
            job = transferred_jobs_queue.peek_job()         # Peek at the job without removing it
            if job and job['computation_request'] <= computational_power:
                transferred_jobs_queue.dequeue_job()        # Now remove the job

                job['processing_time'] = job['computation_request'] / computational_power  # seconds
                computational_power -= job['computation_request']

                # Add the job to the downlink buffer
                accomplished_jobs_queue.enqueue_job(job)

                # Update accomplished_time and mark as accomplished in the packets data frame
                if job['device_type'] == 'user_device':
                    self.job_generator.packet_df_ue.loc[
                        self.job_generator.packet_df_ue['packet_id'] == job['packet_id'], 
                        ['is_accomplished', 'accomplished_time']
                    ] = [True, self.env.time]
                elif job['device_type'] == 'sensor':
                    self.job_generator.packet_df_sensor.loc[
                        self.job_generator.packet_df_sensor['packet_id'] == job['packet_id'], 
                        ['is_accomplished', 'accomplished_time']
                    ] = [True, self.env.time]
                else:
                    logging.warning(f"Unknown device type {job['device_type']}. Computing time not updated.")

                logging.info(
                    f"Time step: {self.env.time} Processed job {job['packet_id']} with computational requirement {job['computation_request']}."
                )

                if computational_power < 0:
                    logging.warning("MEC server computational power exhausted. Some jobs may be delayed.")
                    break  # Exit the loop if computational power is exhausted
            else:
                logging.warning(f"Job {job['packet_id']} requires more computational power than available. Skipping job.")
                break

    def _update_job(self, job: Dict[str, Union[int, float]], bits_to_send: float) -> None:
        job['remaining_request_size'] -= bits_to_send
        if job['remaining_request_size'] < 0:
            job['remaining_request_size'] = 0

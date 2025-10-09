import grpc
from typing import List, Set, Any
from deepspeed.utils import logger
from sailor.protos import orchestration_pb2
from sailor.protos.orchestration_pb2_grpc import WorkerAgentStub


class Controller:
    # A helper class for all the current worker nodes in the cluster.
    # Users can implement their own controller by extending this class
    # We provide a GKE-based controller in the file 'gke_controller.py'

    def __init__(self) -> None:
        self.node_names: Set[str] = set()
        self.node_objs: List[Any] = []
        self.disk_vm_map = {}
        self.curr_vm_list = []

    @staticmethod
    def notify_topology_change(node, configuration, hparams, ip_addr, port, remote_ckpt=None, retry=1):
        request = orchestration_pb2.WorkerConfigurationRequest(
            configuration=configuration,
            hyper_params=hparams,
            remote_ckpt=remote_ckpt
        )
        grpc_target = f'{ip_addr}:{port}'
        success = False
        while True:
            logger.info(f"Sending new topology to {grpc_target}...")
            try:
                with grpc.insecure_channel(grpc_target) as channel:
                    stub = WorkerAgentStub(channel)
                    stub.ConfigurationChange(request)
                    success = True
            except Exception as e:
                logger.error(
                    f"Exception when calling ConfigurationChange: {e}")
                if retry > 0:
                    retry -= 1
                    continue
                raise e
            if success:
                break

    @staticmethod
    def send_kill_request(ip_addr, port, to_exit, retry=3):
        request = orchestration_pb2.KillRequest(exit=to_exit)
        grpc_target = f'{ip_addr}:{port}'

        while True:
            try:
                with grpc.insecure_channel(grpc_target) as channel:
                    stub = WorkerAgentStub(channel)
                    stub.Kill(request)
                    break
            except Exception as e:
                logger.error(
                    f"Exception when calling ConfigurationChange: {e}")
                if retry > 0:
                    retry -= 1
                    continue
                raise e
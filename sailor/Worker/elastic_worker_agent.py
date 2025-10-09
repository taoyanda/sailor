import sys
import grpc
from concurrent import futures
import argparse
import signal
from deepspeed.utils import logger
import traceback
import sailor
from sailor.protos import orchestration_pb2_grpc, orchestration_pb2

from torch import multiprocessing
import threading
import torch
import time
import json
import os
from pathlib import Path


WORKER_AGENT_PORT = "50051"
TRAINING_START_PORT = "51000"

class ElasticWorkerAgent(orchestration_pb2_grpc.WorkerAgentServicer):
    def __init__(self, server, model_name, static_args):
        self.training_processes = []
        self.restart_events = []
        self.cleanup_events = []

        self.arg_dicts = []
        self.manager = multiprocessing.Manager()

        self.server = server

        self.static_args = static_args
        self.model_name = model_name
        self.lock = threading.Lock()
        self.num_restarts = 0
        self.kill_all = False # whether to kill all training processes on reconfigurations

        with open(self.static_args.ds_config_file, 'r') as f:
            ds_dict = json.load(f)
        self.ds_dict = ds_dict

    def _graceful_stop(self):
        print("Stopping server gracefully in 2 seconds...")
        time.sleep(2)
        self.server.stop(grace=5).wait()
        print("Server stopped.")

    def CheckReady(self, request, context):
        logger.info("----- got request from the controller -----")
        return orchestration_pb2.CheckReadyResponse()

    def CheckHealth(self, request, context):
        logger.info("----- got CheckHealth request from the controller -----")
        processes_ok=True
        for process in self.training_processes:
            if not process.is_alive():
                processes_ok=False
                break
        return orchestration_pb2.CheckHealthResponse(processes_ok=processes_ok) # TODO

    def Kill(self, request, context):
        logger.info("----- got kill request from the controller")
        with self.lock:
            if self.kill_all:
                try:
                    for x in self.training_processes:
                        x.terminate()
                except Exception:
                    logger.error(
                        "failed to terminate the training process; here is the traceback:")
                    logger.error(traceback.format_exc())
                else:
                    logger.info("the training process has been terminated")
                finally:
                    self.training_processes = []
            else:
                # if no kill, just reset:
                print(f"just reset, num processes is {len(self.cleanup_events)}")
                for event in self.cleanup_events:
                    event.set()
                # wait for processes to cleanup
                for event in self.cleanup_events:
                    while event.is_set():
                        time.sleep(0.1)

                if request.exit:
                    print(f"Kill all local training processes and cleanup")
                    for x in self.training_processes:
                        x.terminate()
                        x.join()

                    print(f"Starting gracefull termination ....")
                    threading.Thread(target=self._graceful_stop).start()

        return orchestration_pb2.KillResponse()

    def ConfigurationChange(self, request, context):
        configuration = request.configuration
        hyper_params = request.hyper_params
        logger.info(
            f"Cluster change detected, new configuration is {configuration}")
        logger.info(f"Hyper params are {hyper_params}")

        self.ds_dict['train_batch_size'] = hyper_params.global_batch_size
        self.ds_dict['train_micro_batch_size_per_gpu'] = hyper_params.micro_batch_size
        self.ds_dict['gradient_accumulation_steps'] = hyper_params.ga_steps

        with open(self.static_args.ds_config_file, 'w') as f:
            json.dump(self.ds_dict, f, indent=2)

        all_ranks_configs = []
        for stage in configuration.all_stages:
            stage_config = []
            for replica in stage.stage_replicas:
                stage_config.append([int(x) for x in replica.replica_ranks])
            all_ranks_configs.append(stage_config)

        print(f"all_ranks_configs is {all_ranks_configs}")
        with open("dist_config.json", 'w') as f:
            json.dump(all_ranks_configs, f)

        layers_per_stage = json.dumps(list(configuration.layers_per_stage))

        for i,rank in enumerate(configuration.ranks):
            print(f"Start or adapt worker with rank {rank}")
            self.spawn_worker(
                i,
                hyper_params.global_batch_size,
                hyper_params.micro_batch_size,
                hyper_params.num_stages,
                rank,
                configuration.world_size,
                configuration.master_ip,
                configuration.master_port,
                self.static_args.ds_config_file,
                configuration.tensor_parallelism,
                configuration.pipeline_parallelism,
                configuration.data_parallelism,
                configuration.max_tensor_parallelism,
                layers_per_stage
            )

        return orchestration_pb2.WorkerConfigurationResponse()


    def spawn_worker(
        self,
        worker_idx,
        global_batch_size,
        micro_batch_size,
        num_stages,
        rank,
        world_size,
        master_ip,
        master_port,
        ds_config_file,
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        data_parallelism=None,
        max_tensor_parallelism=None,
        layers_per_stage=None
    ):

        sys.path.append("/root/sailor/third_party/Megatron-DeepSpeed")
        from train_llm import run_megatron as run
        from megatron.arguments import parse_args

        training_args = parse_args(extra_args_provider=None, ignore_unknown_args=True)
        training_args.rank = rank
        training_args.world_size = world_size
        training_args.master_ip = master_ip
        training_args.master_port = master_port
        training_args.global_batch_size = global_batch_size
        training_args.micro_batch_size = micro_batch_size
        if tensor_model_parallel_size is not None:
            training_args.tensor_model_parallel_size = tensor_model_parallel_size
            training_args.pipeline_model_parallel_size = pipeline_model_parallel_size
            training_args.data_parallelism = data_parallelism
            training_args.max_tensor_parallelism = max_tensor_parallelism
            training_args.distributed_config_file = "dist_config.json"
            training_args.layers_per_stage = layers_per_stage
        training_args.deepspeed_config = ds_config_file

        with self.lock:
            if len(self.training_processes) < worker_idx + 1:
                # start a new training process
                self.restart_events.append(multiprocessing.Event())
                self.cleanup_events.append(multiprocessing.Event())

                arg_dict = self.manager.dict()
                arg_dict['args'] = training_args
                self.arg_dicts.append(arg_dict)

                logger.info("starting a new training process...")
                new_proc = multiprocessing.Process(
                    target=run, args=(training_args, self.cleanup_events[worker_idx], self.restart_events[worker_idx], self.arg_dicts[worker_idx]))
                new_proc.start()
                self.training_processes.append(new_proc)
                logger.info("a new training process has started")
            else:
                logger.info("adapting existing training process...")
                self.arg_dicts[worker_idx]['args'] = training_args
                self.restart_events[worker_idx].set()



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # pre-download datasets
    from datasets import load_dataset
    load_dataset('wikitext', 'wikitext-103-v1', split='train')

    # We need to parse the model-specific arguments first to determine the model name
    pre_parser = argparse.ArgumentParser(description='parser for static args')
    pre_parser.add_argument('--model_name', type=str)
    choice_parse = pre_parser.parse_known_args()[0]
    model_name = choice_parse.model_name
    parser = argparse.ArgumentParser(description='parser for static args')
    parser.add_argument('--agent_port', type=int, help='Port for gRPC agent', default=WORKER_AGENT_PORT)
    parser.add_argument('--bucket_name', type=str,
                        help='the name of the google cloud storage bucket')
    parser.add_argument('--remote_root_dir', type=str,
                        help='the root directory of the remote storage for checkpoints')
    parser.add_argument('--local_root_dir', type=str,
                        help='the root directory for the local checkpoints')
    parser.add_argument('--num_iters', type=int, default=0,
                        help='number of iterations to train in total; 0 to run infinitely')
    parser.add_argument('--log_interval', type=int,
                        default=20, help='log every n steps')
    parser.add_argument('--mixed_precision_training', default=False, action='store_true',
                        help='whether to use mixed precision training')
    parser.add_argument('--with_controller', default=False, action='store_true',
                        help='run with controller')
    parser.add_argument('--ds_config_file', type=str,
                        help='DeepSpeed config file', required=True)

    static_user_args = parser.parse_known_args()[0]

    logs_dir = "/root/sailor/third_party/Megatron-DeepSpeed/logs"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    os.environ['SAILOR_LOGS_DIR'] = logs_dir

    # do compilation of fused kernels, to save time
    os.system("cd /root/sailor/third_party/Megatron-DeepSpeed/ && python do_load.py")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent = ElasticWorkerAgent(server, model_name, static_user_args)
    orchestration_pb2_grpc.add_WorkerAgentServicer_to_server(agent, server)
    server.add_insecure_port(f'[::]:{static_user_args.agent_port}')

    def terminate(signum, _):
        if agent.training_process is not None:
            try:
                agent.training_process.terminate()
            except Exception:
                logger.error(
                    "failed to terminate the training process; here is the traceback:")
                logger.error(traceback.format_exc())
            else:
                logger.info("the training process has been terminated")
            finally:
                agent.training_process = None
        done = server.stop(5)
        done.wait()
        logger.info(f"Received {signum}, stop complete!")

    logger.info("starting server")
    server.start()
    #signal.signal(signal.SIGTERM, terminate)
    server.wait_for_termination()

    logger.info("server closed")
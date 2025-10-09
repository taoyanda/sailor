"""Process for asynchronous checkpointing"""

import torch
import os
import copy
import time
from collections import OrderedDict
import signal
import sys
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Barrier, Event


from sailor.Worker.worker_utils import debug_print
from sailor.Worker.checkpoint.chk_writer import ChkWriter

class Chk_manager:

    def __init__(self, save_dir, start_layer, rank, pp_rank, tp_rank):

        # TODO: replace values with events for better perf
        manager = Manager()
        self.res_state = manager.dict()
        self.lock = Lock()
        self.cp_in_progress = Event()
        self.start = Event()
        self.stop = Event()
        self.set = Event()
        self.initialized = False
        self.barrier = Barrier(2)
        self.save_dir = save_dir
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.rank = rank
        self.start_layer = start_layer
        self.chk_process = None

    def init_buffer_and_writer(self, model, optimizer):

        # self.model_state.update(model.state_dict())
        # self.opt_state.update(optimizer.state_dict())

        self.chk_writer = ChkWriter(self.save_dir, self.start_layer, self.rank, self.tp_rank)
        self.chk_process = Process(
            target=self.chk_writer.save_async_torch,
            args=[model, optimizer, self.res_state, self.lock, self.set, self.cp_in_progress, self.start, self.stop, self.barrier],
            daemon=True
        )
        self.chk_process.start()

        self.set.set()

        self.barrier.wait()
        self.initialized = True

    def gpu_copy_in_progress(self):

        # return True if at least one of the background processes is copying
        if self.cp_in_progress.is_set():
            return True

        return False

    def checkpoint_in_progress(self):
        if self.start.is_set():
            return True

    def save_checkpoint(self, res_dict):
        while True:
            if not self.start.is_set():
                break

        # additional state here
        self.res_state.update(res_dict)

        with self.lock:
            self.cp_in_progress.set()
            self.start.set()


    def kill_checkpoint(self):

        # graceful termination - wait for current checkpoint to finish
        while True:
            with self.lock:
                if not self.start.is_set():
                    break

        self.stop.set()
        self.start.set()

        if self.chk_process:
            self.chk_process.join()

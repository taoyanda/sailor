"""Process for asynchronous checkpointing"""

import torch
import os
import copy
import time
from collections import OrderedDict
import signal
import sys
from ctypes import *
import numpy as np

from sailor.Worker.worker_utils import debug_print
from concurrent.futures import ThreadPoolExecutor, wait


class ChkWriter:

    def __init__(self, save_dir, start_layer, rank, tp_rank):

        self.cpu_buffer = None
        self.model = None
        self.optimizer = None
        self.save_dir = save_dir
        self.tp_rank = tp_rank
        self.rank = rank
        self.start_layer = start_layer
        self.index = 0

    def init_buffer(self):
        # TODO: fix for fp16
        sz = 0
        for i,module in enumerate(self.model):
            if not hasattr(module, 'state_dict'):
                 continue
            if hasattr(module, 'parameters'):
                 for k, v in module.state_dict().items():
                     if torch.is_tensor(v):
                         sz += torch.numel(v)
                 for name, p in module.named_parameters():
                     if not ('weight' in name or 'bias' in name):
                         continue
                     for _, val in self.optimizer.state[p].items():
                         sz += torch.numel(val)
        self.cpu_buffer=torch.empty(sz, dtype=torch.float32, pin_memory=True, device="cpu")
        self.copy_stream = torch.cuda.Stream()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def save_async_torch(
        self,
        model,
        optimizer,
        res_state,
        lock,
        initial_set,
        cp_in_progress,
        start,
        stop,
        barrier
    ):
        # simple, torch-based, unoptimized implementation
        print(f"************ Async proc started")

        # wait to be initialized
        while True:
            with lock:
                if initial_set.is_set():
                    break

        self.model = model
        self.optimizer = optimizer
        self.res_state = res_state

        self.init_buffer()
        barrier.wait()

        print(f"************ About to enter checkp-train loop")
        # checkp-train loop
        while True:

            start.wait()
            if stop.is_set():
                break

            self.save_torch(self.res_state['global_steps'])

            with lock:
                cp_in_progress.clear()
                start.clear()

        # exit
        self.executor.shutdown()

    def save_numpy(self, start, end, key):
        self.cpu_buffer[start:end].detach().numpy().tofile(f"{self.save_dir}/module_{key}_{self.tp_rank}_index_{self.index}")

    def save_torch(self, iteration):
        print(f"SAVE, Iteration is {iteration}, TP RANK is {self.tp_rank}")
        idx = 0
        futures = []
        if True: #with ThreadPoolExecutor(max_workers=4) as executor:
            for i,module in enumerate(self.model):
              prev_idx = idx
              key = i+self.start_layer
              if not hasattr(module, 'state_dict'):
                  continue
              if hasattr(module, 'parameters'):
                   for _, value in module.state_dict().items():
                        if torch.is_tensor(value):
                            sz = torch.numel(value)
                            self.cpu_buffer[idx:idx+sz].copy_(value.flatten())
                            idx+=sz
                   for name, p in module.named_parameters():
                       if not ('weight' in name or 'bias' in name):
                            continue
                       for opt_key, val in self.optimizer.state[p].items():
                            sz = torch.numel(val)
                            with torch.cuda.stream(self.copy_stream):
                                self.cpu_buffer[idx:idx+sz].copy_(val.flatten()) # many small transfers, could be optimized by batching them
                            idx+=sz
                   torch.cuda.synchronize()
                   futures.append(self.executor.submit(self.save_numpy, prev_idx, idx, key))
        #torch.distributed.barrier() # NOTE: this might be quite slow
        wait(futures)
        self.index = 1-self.index
        if self.rank==0:
            deepspeed_state = self.res_state.copy()
            deepspeed_state["index"] = self.index
            torch.save(deepspeed_state, f"{self.save_dir}/check_{iteration}")

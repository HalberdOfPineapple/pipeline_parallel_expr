import torch
from torch import nn

from . import stream_utils, microbatch
from .checkpointing import CheckPointingCLS, Join, CheckPointingCLSv2, join
from .microbatch import Batch
from .dependency import depend

from collections import OrderedDict
from typing import List, Tuple, Callable, Any, Dict, Iterable


ALWAYS = 'always'
EXCEPT_LAST = 'except_last'
NEVER = 'never'

def print_mem_usage(device: torch.device):
    memory_usage = torch.cuda.memory_reserved(device)
    print(f'{device!s}: {memory_usage:,} Bytes ({memory_usage / 1024**3:.2f} GiB)')

def print_devices_mem_usage(devices: List[torch.device]):
    for device in devices:
        print_mem_usage(device)


def split_module(
      module: torch.nn.Module, 
      partition_sizes: List[int], 
      devices: List[torch.device]
    ) -> Tuple[nn.ModuleList, List[torch.device]]:
    print('-' * 80)
    print('Splitting the module in MyGPipe...')
    layers = OrderedDict()
    partitions = []

    i = 0
    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == partition_sizes[i]:
            partitions.append(nn.Sequential(layers).to(devices[i]))
            print_mem_usage(devices[i])

            layers.clear()
            i += 1
    del devices[i:]
    return torch.nn.ModuleList(partitions), devices


def schedule_tasks(num_micro_batches, num_partitions):
    schedules = []
    for step in range(num_micro_batches + num_partitions - 1):
        schedule = []
        for partition_id in range(
            max(0, step + 1 - num_micro_batches),
            min(step + 1, num_partitions)
        ):
            # Add (micro_batch_id, partition_id)
            schedule.append((step - partition_id, partition_id))
        schedules.append(schedule)
    return schedules


class GPipe(torch.nn.Module):
    def __init__(
        self, 
        module: nn.Sequential, 
        partition_plan: List[int], 
        num_micro_batches: int,
        devices: List[int] | List[torch.device]=None,
        checkpoint_strategy: str='except_last',
    ):
        """
        Args:
            module: the module to be parallelized
            partition_plan: a list of integers, each integer represents the number of layers in each partition
            num_micro_batches: the number of microbatches
            devices: a list of devices, each device represents the device where each partition is placed
        """
        super().__init__()

        self.num_micro_batches = num_micro_batches

        if devices is None:
            if torch.cuda.is_available():
                devices = [torch.device('cuda:{}'.format(i % torch.cuda.device_count())) for i in range(len(partition_plan))]
            else:
                devices = [torch.device('cpu') for _ in partition_plan]
        elif isinstance(devices[0], int):
            devices = [torch.device('cuda:{}'.format(device)) for device in devices]
        self.devices = devices

        self.partition_plan = partition_plan
        if sum(partition_plan) != len(module):
            raise ValueError('Module and sum of partition plan have different length')
        self.num_partitions = len(partition_plan)
        self.partitions, self.devices = split_module(module, partition_plan, self.devices)

        self.init_checkpoint_layers(checkpoint_strategy)
        self.init_streams()

    def init_checkpoint_layers(self, checkpoint_strategy: str):
        """Setup the layers to be checkpointed depending on the checkpoint strategy"""

        checkpoint_strategy = checkpoint_strategy if checkpoint_strategy else NEVER
        if checkpoint_strategy == ALWAYS:
            self.checkpoint_layers = list(range(len(self.partitions)))
        elif checkpoint_strategy == EXCEPT_LAST:
            self.checkpoint_layers = list(range(len(self.partitions) - 1))
        elif checkpoint_strategy == NEVER:
            self.checkpoint_layers = []
        else:
            raise ValueError(f'Unsupported checkpoint strategy: {checkpoint_strategy}')
    
        self.checkpoint_stop = {
            'always': self.num_micro_batches,
            'except_last': self.num_micro_batches,
            'never': 0,
        }[checkpoint_strategy]
    
    def init_streams(self):
        """Initialize streams for each partition"""
        self.copy_streams = [
            [stream_utils.init_stream(device) for _ in range(self.num_micro_batches)]
            for device in self.devices
        ]
        self.cal_streams = [stream_utils.get_default_stream(device) for device in self.devices]

    def forward(self, input_data: torch.Tensor):
        # list of tensors, each of which has shape (micro_batch_size, ...)
        micro_batches: List[Batch] = microbatch.split_batch(input_data, self.num_micro_batches)


        # Note that number of microbatches may be different from pre-defined self.num_micro_batches
        # e.g. when mini-batch size (1) is smaller than number of microbatches (4)
        schedules: List[List[Tuple[int, int]]] = schedule_tasks(
                                        num_micro_batches=len(micro_batches),
                                        num_partitions=self.num_partitions,)

        for i, schedule in enumerate(schedules):
            for micro_batch_id, partition_id in schedule:
                # print('-' * 30)
                # print(f"Executing Step {i}, Micro-batch {micro_batch_id}, Partition {partition_id}")
                micro_batch: Batch = micro_batches[micro_batch_id]
                if micro_batch_id > 0:
                    depend(micro_batches[micro_batch_id - 1], micro_batch)

                if partition_id > 0:
                    micro_batch = stream_utils.stream_copy(
                        self.copy_streams[partition_id - 1][micro_batch_id],
                        self.copy_streams[partition_id][micro_batch_id],
                        micro_batch,
                    )

                    # Current calculation stream waiting for the copy being finished
                    micro_batch = stream_utils.stream_wait(
                        self.cal_streams[partition_id],
                        self.copy_streams[partition_id][micro_batch_id],
                        micro_batch
                    )

                # Forward Calculation of this microbatch
                ckpt = None
                if micro_batch_id < self.checkpoint_stop:
                    ckpt = CheckPointingCLSv2(
                        self.cal_streams[partition_id], 
                        self.partitions[partition_id],
                        micro_batch,
                    )
                    micro_batch = ckpt.checkpoint()
                else:
                    with torch.cuda.stream(self.cal_streams[partition_id]):
                        micro_batch = micro_batch.call(self.partitions[partition_id])

                # For non-last partitions, the copy stream should wait for the calculation stream to finish before copying data to the next partition
                if partition_id < self.num_partitions - 1:
                    micro_batch = stream_utils.stream_wait(
                        self.copy_streams[partition_id][micro_batch_id],
                        self.cal_streams[partition_id],
                        micro_batch
                    )

                # Recompute
                if micro_batch_id < self.checkpoint_stop:
                    ckpt.recompute(micro_batch)

                # Update the data
                micro_batches[micro_batch_id] = micro_batch
    
        output = microbatch.merge_data(micro_batches)
        return output
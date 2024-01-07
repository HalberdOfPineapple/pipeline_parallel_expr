import torch
from torch import nn
import stream_utils

from checkpointing import CheckPointingCLS, Join
from collections import OrderedDict
from typing import List, Tuple, Callable, Any, Dict, Iterable


ALWAYS = 'always'
EXCEPT_LAST = 'except_last'
NEVER = 'never'

def split_module(
      module: torch.nn.Module, 
      partition_sizes: List[int], 
      devices: List[torch.device]
    ) -> nn.ModuleList:
    layers = OrderedDict()
    partitions = []

    i = 0
    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == partition_sizes[i]:
            partitions.append(nn.Sequential(layers).to(devices[i]))
            layers.clear()
            i += 1
    
    return torch.nn.ModuleList(partitions)

def split_batch(batch: torch.Tensor, num_micro_batches: int) -> List[torch.Tensor]:
    """Split a batch into a list of microbatches"""
    if isinstance(batch, torch.Tensor):
        split_batch = batch.chunk(num_micro_batches)
    else: # batch is a list of tensors
        split_batch = []
        for tensor in batch:
            split_tensor = tensor.chunk(num_micro_batches)
            split_batch.append(split_tensor)
        split_batch = zip(*split_batch)
    return list(split_batch)

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

def merge_data(
        micro_batches: List[torch.Tensor] | Iterable[Iterable[torch.Tensor]],
    ) -> torch.Tensor | Tuple[torch.Tensor]:
    if isinstance(micro_batches[0], torch.Tensor):
        return torch.cat([micro_batch for micro_batch in micro_batches])
    else:
        micro_batches = [micro_batch for micro_batch in micro_batches]
        merged_micro_batches = []
        for tensors in zip(*micro_batches):
            merged_micro_batches.append(torch.cat(tensors))
        return tuple(merged_micro_batches)

class GPipe(torch.nn.Module):
    def __init__(
        self, 
        module: nn.Module, 
        partition_plan: List[int], 
        num_micro_batches: int,
        devices: List[torch.device]=None,
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
        self.devices = devices

        self.num_partitions = len(partition_plan)
        self.partitions = split_module(module, partition_plan, devices)

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
    
    def init_streams(self):
        """Initialize streams for each partition"""
        self.copy_streams = [
            [stream_utils.init_stream(device) for _ in range(self.num_micro_batches)]
            for device in self.devices
        ]
        self.cal_streams = [stream_utils.get_default_stream(device) for device in self.devices]
    
    def split_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Split a batch into a list of microbatches"""
        return split_batch(batch, self.num_micro_batches)

    def forward(self, input_data: torch.Tensor):
        # list of tensors, each of which has shape (micro_batch_size, ...)
        micro_batches: List[torch.Tensor] = self.split_batch(input_data)

        # Note that number of microbatches may be different from pre-defined self.num_micro_batches
        # e.g. when mini-batch size (1) is smaller than number of microbatches (4)
        schedules: List[List[Tuple[int, int]]] = schedule_tasks(
                                        num_micro_batches=len(micro_batches),
                                        num_partitions=self.num_partitions,)

        for schedule in schedules:
            for micro_batch_id, partition_id in schedule:
                micro_batch = micro_batches[micro_batch_id]

                # For non-first partitions, wait for the previous partition copying data to the current partition before executing
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
                if partition_id not in self.checkpoint_layers:
                    micro_batch = self.partitions[partition_id](micro_batch)
                else:
                    if partition_id in self.checkpoint_layers:
                        ckpt = CheckPointingCLS(
                            self.cal_streams[partition_id], 
                            self.partitions[partition_id],
                            micro_batch,
                        )
                        micro_batch = ckpt.checkpoint()

                # For non-last partitions, the copy stream should wait for the calculation stream to finish before copying data to the next partition
                if partition_id < self.num_partitions - 1:
                    micro_batch = stream_utils.stream_wait(
                        self.copy_streams[partition_id][micro_batch_id],
                        self.cal_streams[partition_id],
                        micro_batch
                    )
                
                # Recompute
                if partition_id in self.checkpoint_layers:
                    aux_tensor = ckpt.recompute(micro_batch)
                    micro_batch = Join.apply(micro_batch, aux_tensor)

                # Update the data
                micro_batches[micro_batch_id] = micro_batch

        # for micro_batch in micro_batches:
        #     print(micro_batch.device)
    
        output = merge_data(micro_batches)
        return output
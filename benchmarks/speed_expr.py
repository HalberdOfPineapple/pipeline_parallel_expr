"""AmoebaNet-D (L, D) Memory Benchmark"""
import platform
import argparse
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import RMSprop

from amoebanet import amoebanetd
from resnet import resnet100, build_train_stuffs

import torchgpipe
from torchgpipe import GPipe
from MyGPipe import GPipe as SelfGPipe
from utils import init_logger, get_logger, print_log, DATA_DIR

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

Stuffs = Tuple[nn.Module, int, int, List[torch.device]]  # (model, L, D, devices)
Experiment = Callable[[List[int]], Stuffs]


def hr():
    print_log('-' * 80)

# 3 + [3, 13, 30, 3] + 4
# 3 x 64 + 13 x 128 + 30 x 256 + 3 x 512
partition_plans: Dict[str, List[int]] = {
    1: [56], 
    2: [33, 23],
    4: [15, 13, 14, 14],
}

batch_sizes: Dict[str, int] = {
    'n1m1': 64,
    'n1m4': 256,
    'n1m32': 2048,
    'n2m1': 128,
    'n2m4': 512,
    'n2m32': 4096,
    'n4m1': 256,
    'n4m4': 1024,
    'n4m32': 8192,
}

def args_to_exprname(args: Dict[str, Any]) -> str:
    return (
        f'speed_K{args["num_partitions"]}_M{args["num_micro_batches"]}_'
        f'{"torchgpipe" if args["use_torchgpipe"] else "self"}')

def get_batch_size(args: Dict[str, Any]) -> int:
    return batch_sizes[f'n{args["num_partitions"]}m{args["num_micro_batches"]}']

# baseline - (K=1, M=1)
def init_expr(config, model: nn.Sequential, partition_plans: Dict[int, List[int]]):
    num_partitions = config['num_partitions']
    expr_name =  args_to_exprname(config)
    init_logger(expr_name, speed_log=True)

    print_log('=' * 80 + '\nConfiguration')
    for k, v in config.items():
        print_log(f'{k}: {v}')
    print_log('=' * 80)

    parititon_plan = partition_plans[num_partitions]
    num_microbatches = config['num_micro_batches']
    checkpoint_strategy = 'except_last' if config['checkpoint_enabled'] else 'never'

    hr()
    use_torchgpipe = config['use_torchgpipe']
    print_log(f'Pipelining model with {"self-implemented GPipe" if not use_torchgpipe else "torchgpipe"}...')
    if use_torchgpipe:
        model = GPipe(model, balance=parititon_plan, chunks=num_microbatches, checkpoint=checkpoint_strategy)
    else:
        model = SelfGPipe(
            module=model, 
            partition_plan=parititon_plan,
            num_micro_batches=num_microbatches,
            checkpoint_strategy=checkpoint_strategy,
        )

    devices = model.devices
    return model, devices

def main(config):
    print_log("Running ResNet expr...")

    num_classes = 10
    model = resnet100(num_classes=num_classes)
    model = cast(nn.Sequential, model)
    model, devices = init_expr(config, model, partition_plans)

    # ============================================================================
    # Training Related
    num_batches = 10
    batch_size = get_batch_size(config)
    train_loader, criterion, optimizer = build_train_stuffs(
        model, batch_size, DATA_DIR)
    input_device, output_device = devices[0], devices[-1]

    print_log(f"Number of batches: {num_batches}")
    print_log(f"Batch size: {batch_size}")

    # ============================================================================
    # Profiling by training
    hr()
    print_log("Profiling...")
    
    # Prepare statistics for profiling
    model.train()
    torch.cuda.empty_cache()
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)

    iter_times = []
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == num_batches: break
        inputs, targets = inputs.to(input_device), targets.to(output_device)

        start_time = perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        

        start_time = perf_counter()
        optimizer.zero_grad()
        loss.backward()
        iter_times.append(perf_counter() - start_time)

        # Optimizer step is outside of the backward pass timing
        optimizer.step() 
        print_log(f"Iter {idx} time: {iter_times[-1]:.3f}")
    total_time = sum(iter_times)

    print_log(f"Throughput: {batch_size * num_batches / total_time:.3f} samples/sec")
    print_log("Profiling Completed.")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_partitions', '-k', type=int, default=2)
    argparser.add_argument('--num_micro_batches', '-m', type=int, default=1)
    argparser.add_argument('--checkpoint_enabled', '-c', action='store_true')
    argparser.add_argument('--use_torchgpipe', '-t', action='store_true') # default: self-gpipe
    config = argparser.parse_args()

    main(vars(config))

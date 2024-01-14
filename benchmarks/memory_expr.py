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

L, D = 18, 416
amoeba_partition_plans: Dict[str, List[int]] = {
    0: [24],
    1: [24],
    2: [16, 8],
    4: [12, 6, 6],
}

# 3 + [3, 13, 30, 3] + 4
# 3 x 64 + 13 x 128 + 30 x 256 + 3 x 512
resnet100_partition_plans: Dict[str, List[int]] = {
    1: [56], 
    2: [33, 23],
    4: [15, 13, 14, 14],
}

def init_expr(config, model: nn.Sequential, partition_plans: Dict[int, List[int]]):
    num_partitions = config['num_partitions']
    if num_partitions == 0:
        print_log("Running the baseline...")
        expr_name = "baseline"
        init_logger(expr_name)

        devices = [torch.device('cuda:0')]
        model.to(devices[0])
    else:
        expr_name = (
        f'mem_expr_K{config["num_partitions"]}_M{config["num_microbatches"]}_'
        f'{"check_" if config["checkpoint_enabled"] else ""}'
        f'{"torchgpipe" if  config["use_torchgpipe"] else "self"}')
        init_logger(expr_name)

        print_log('=' * 80 + '\nConfiguration')
        for k, v in config.items():
            print_log(f'{k}: {v}')
        print_log('=' * 80)

        parititon_plan = partition_plans[num_partitions]
        num_microbatches = config['num_microbatches']
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

def profile_params(model: nn.Sequential, param_scale: int):
    hr()
    print_log("Profiling parameters...")
    param_count = sum(p.storage().size() for p in model.parameters())
    param_size = sum(p.storage().size() * p.storage().element_size() for p in model.parameters())

    print_log(f'# of Model Parameters: {param_count:,}')
    print_log(f'Total Model Parameter Memory: {param_size * param_scale:,} Bytes ({param_size * param_scale / 1024**3:.2f} GiB)')
    return param_count, param_size

def print_max_memory(devices: List[torch.device]):
    hr()
    print_log("Max memory per device:")
    for d in devices:
        memory_usage = torch.cuda.memory_reserved(d)
        print_log(f'{d!s}: {memory_usage:,} Bytes ({memory_usage / 1024**2:.2f} MiB)')

def run_resnet_expr(config):
    print_log("Running ResNet expr...")

    num_classes = 10
    model = resnet100(num_classes=num_classes)
    model = cast(nn.Sequential, model)
    model, devices = init_expr(config, model, resnet100_partition_plans)

    # ============================================================================
    # Training Related
    num_batches = 10
    batch_size = 128
    train_loader, criterion, optimizer = build_train_stuffs(
        model, batch_size, DATA_DIR)
    input_device, output_device = devices[0], devices[-1]

    # =====================================================================
    # Parameters
    param_scale = 3
    param_count, param_size = profile_params(model, param_scale=3)

    # ============================================================================
    # Profiling by training
    hr()
    print_log("Profiling...")
    
    # Prepare statistics for profiling
    model.train()
    torch.cuda.empty_cache()
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []

    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == num_batches: break
        inputs, targets = inputs.to(input_device), targets.to(output_device)

        start_time = perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        forward_times.append(perf_counter() - start_time)

        start_time = perf_counter()
        optimizer.zero_grad()
        loss.backward()
        backward_times.append(perf_counter() - start_time)

        # Optimizer step is outside of the backward pass timing
        optimizer.step() 
    total_time = sum(forward_times) + sum(backward_times)

    # Save memory profiling statistics
    hr()
    max_memory = 0
    for d in devices:
        torch.cuda.synchronize(d)
        max_memory += torch.cuda.max_memory_reserved(d)
    latent_size = max_memory - param_size * param_scale

    print_log(f'Peak Activation Memory: {latent_size:,} Bytes ({latent_size / 1024**2:.2f} MiB)')
    print_log(f'Total Memory: {max_memory:,} Bytes ({max_memory / 1024**2:.2f} MiB)')
    print_log(f'Average Forward Time: {sum(forward_times)/len(forward_times):.3f} sec')
    print_log(f'Average Backward Time: {sum(backward_times)/len(backward_times):.3f} sec')
    print_log(f"Throughput: {batch_size * num_batches / total_time:.3f} samples/sec")
    print_log("Profiling Completed.")

    # ============================================================================
    # Max memory per device
    print_max_memory(devices)


def main(config):
    if config['use_resnet']:
        run_resnet_expr(config)
        return

    model = amoebanetd(num_classes=1000, num_layers=L, num_filters=D)
    model = cast(nn.Sequential, model)
    model, devices = init_expr(config, model, amoeba_partition_plans)

    # =====================================================================
    # Training-related
    optimizer = RMSprop(model.parameters())

    in_device = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    num_samples = 5
    batch_size = 128
    num_classes = 1000
    input = torch.rand(batch_size, 3, 224, 224, device=in_device)
    target = torch.randint(num_classes, (batch_size,), device=out_device)

    # =====================================================================
    # Parameters
    param_scale = 3
    param_count, param_size = profile_params(model, param_scale)

    # =====================================================================
    # Profiling
    hr()
    print_log("Start Profiling...")
    torch.cuda.empty_cache()
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)

    forward_times, backward_times = [], []
    for _ in range(num_samples):
        start_time = perf_counter()
        output = model(input)
        forward_times.append(perf_counter() - start_time)

        loss = F.cross_entropy(cast(Tensor, output), target)

        start_time = perf_counter()
        loss.backward()
        backward_times.append(perf_counter() - start_time)

        optimizer.step()

    max_memory = 0
    for device in devices:
        torch.cuda.synchronize(device)
        max_memory += torch.cuda.max_memory_reserved(device)

    latent_size = max_memory - param_size * param_scale
    print_log(f'Peak Activation Memory: {latent_size:,} Bytes ({latent_size / 1024**3:.2f} GiB)')
    print_log(f'Total Memory: {max_memory:,} Bytes ({max_memory / 1024**3:.2f} GiB)')
    print_log(f'Average Forward Time: {sum(forward_times)/len(forward_times):.3f} sec')
    print_log(f'Average Backward Time: {sum(backward_times)/len(backward_times):.3f} sec')
    print_log("Profiling Completed.")

    # MAX MEMORY PER DEVICE =======================================================================
    hr()
    for device in devices:
        memory_usage = torch.cuda.memory_reserved(device)
        print_log(f'{device!s}: {memory_usage:,} Bytes ({memory_usage / 1024**3:.2f} GiB)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_partitions', '-k', type=int, default=2)
    argparser.add_argument('--num_microbatches', '-m', type=int, default=128)
    argparser.add_argument('--checkpoint_enabled', '-c', action='store_true')
    argparser.add_argument('--use_torchgpipe', '-t', action='store_true') # default: self-gpipe
    argparser.add_argument('--use_resnet', '-r', action='store_true')
    config = argparser.parse_args()

    main(vars(config))

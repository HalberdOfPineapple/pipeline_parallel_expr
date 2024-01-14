import os
import platform
import argparse
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import RMSprop

from resnet import resnet100, build_train_stuffs
from gpt_2 import gpt2_small, load_text_from_file
from gpt_2 import build_train_stuffs as build_gpt2_stuffs 

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

def args_to_expr_name(model_type: str, args: Dict[str, Any]) -> str:
    return (f'{model_type}_K{args["num_partitions"]}_M{args["num_microbatches"]}_'
           f'{"check_" if args["checkpoint_enabled"] else ""}'
           f'{"torchgpipe" if args["use_torchgpipe"] else "self"}'
    )

def init_expr(config, model: nn.Sequential, partition_plans: Dict[int, List[int]]):
    num_partitions = config['num_partitions']
    model_type = 'resnet' if config['use_resnet'] else 'gpt2'
    if num_partitions == 0:
        print_log("Running the baseline...")
        expr_name = f"{model_type}_baseline"
        init_logger(expr_name, gpt_log=not config['use_resnet'])

        devices = [torch.device('cuda:0')]
        model.to(devices[0])
    else:
        expr_name = args_to_expr_name(model_type, config)
        init_logger(expr_name, gpt_log=not config['use_resnet'])

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

def print_mem_usage(device: torch.device):
    memory_usage = torch.cuda.memory_reserved(device)
    print(f'{device!s}: {memory_usage:,} Bytes ({memory_usage / 1024**3:.2f} GiB)')

def print_devices_mem_usage(devices: List[torch.device]):
    for device in devices:
        print_mem_usage(device)

# 2 + 12 + 2
gpt2_partition_plans: Dict[str, List[int]] = {
    1: [16], 
    2: [8, 8],
    4: [5, 3, 3, 5],
}
def run_gpt2_expr(config):
    # ============================================================================
    # Model related
    num_batches = 10
    seq_length = 128
    batch_size = 64
    vocab_size = 3000
    config['batch_size'] = batch_size

    model = gpt2_small(vocab_size, seq_length)
    model = cast(nn.Sequential, model)
    model, devices = init_expr(config, model, gpt2_partition_plans)
    print_log("Running GPT-2 expr...")

    # ============================================================================
    # Training Related
    
    def load_dataset(batch_size, seq_length):
        num_samples = 2000  # Number of samples in the dataset
        # Create random data samples
        data = torch.randint(0, vocab_size, (num_samples, seq_length))
        # Convert to batches
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    dataloader = load_dataset(batch_size, seq_length)
    criterion = nn.CrossEntropyLoss()  # Common choice for language models
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    input_device, output_device = devices[0], devices[-1]

    # =====================================================================
    # Parameters
    param_scale = 3
    param_count, param_size = profile_params(model, param_scale=3)
    
    # ============================================================================
    # Profiling by training
    hr()
    print_log("Profiling...")
    model.train()
    torch.cuda.empty_cache()

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []

    for batch_idx, (data_batch,) in enumerate(dataloader):
        if batch_idx >= num_batches: break

        # Prepare data (assuming language modeling task)
        inputs = data_batch.to(input_device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_length, )).to(output_device) # (batch_size * seq_length, )

        start_time = perf_counter()
        # Forward pass
        outputs = model(inputs) # (batch_size, seq_length, vocab_size)
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_length, vocab_size)

        loss = criterion(outputs, targets)
        forward_times.append(perf_counter() - start_time)

        # Backward pass and optimization
        optimizer.zero_grad()
        start_time = perf_counter()
        loss.backward()
        backward_times.append(perf_counter() - start_time)

        optimizer.step()
    total_time = sum(forward_times) + sum(backward_times)

    # ============================================================================
    # Save memory profiling statistics
    hr()
    max_memory = 0
    for d in devices:
        torch.cuda.synchronize(d)
        max_memory += torch.cuda.max_memory_reserved(d)
    latent_size = max_memory - param_size * param_scale

    print_log(f'Peak Activation Memory: {latent_size:,} Bytes ({latent_size / 1024**3:.2f} GiB)')
    print_log(f'Total Memory: {max_memory:,} Bytes ({max_memory / 1024**3:.2f} GiB)')
    print_log(f'Average Forward Time: {sum(forward_times)/len(forward_times):.3f} sec')
    print_log(f'Average Backward Time: {sum(backward_times)/len(backward_times):.3f} sec')
    print_log(f"Throughput: {batch_size * num_batches / total_time:.3f} samples/sec")
    print_log("Profiling Completed.")

    # ============================================================================
    # Max memory per device
    print_max_memory(devices)
    


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
    else:
        run_gpt2_expr(config)
        return

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_partitions', '-k', type=int, default=2)
    argparser.add_argument('--num_microbatches', '-m', type=int, default=128)
    argparser.add_argument('--checkpoint_enabled', '-c', action='store_true')
    argparser.add_argument('--use_torchgpipe', '-t', action='store_true') # default: self-gpipe
    argparser.add_argument('--use_resnet', '-r', action='store_true')
    config = argparser.parse_args()

    main(vars(config))

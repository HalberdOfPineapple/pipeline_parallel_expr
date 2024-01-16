"""AmoebaNet-D (18, 256) Speed Benchmark"""
import platform
import argparse
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data

from amoebanet import amoebanetd
import torchgpipe
from torchgpipe import GPipe
from MyGPipe import GPipe as SelfGPipe
from utils import init_logger, get_logger, print_log, DATA_DIR

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]

def pipeline_model(model: nn.Module,
           devices: List[int],
           chunks: int,
           balance: List[int],
           checkpoint: str,
           use_self_gpipe: bool):
    if not use_self_gpipe:
        print("Using torchgpipe...")
        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpoint)
    else:
        print_log("Using self-implemented Gpipe...")
        model = cast(nn.Sequential, model)
        model = SelfGPipe(
                    model, 
                    partition_plan=balance, 
                    devices=devices, 
                    num_micro_batches=chunks, 
                    checkpoint_strategy=checkpoint)
    return model, list(model.devices)

EXPERIMENTS: Dict[str, Tuple] = {
    'n2m1': (155, 1, [7, 17]),
    'n2m4': (480, 4, [9, 15]),
    'n2m32': (1500, 32, [9, 15]),
    'n4m1': (260, 1, [3, 4, 5, 12]),
    'n4m4': (600, 4, [3, 6, 7, 8]),
    'n4m32': (980, 32, [3, 6, 7, 8]),
}


BASE_TIME: float = 0


def hr():
    print_log('-' * 80)


def main(args) -> None:
    epochs = args.epochs
    skip_epochs = args.skip_epochs
    if skip_epochs >= epochs:
        raise ValueError('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    # ============================================================================
    # Model related
    experiment = args.experiment
    init_logger(experiment, speed_log=True)

    num_classes = 1000
    num_layers = 18
    num_filters = 256
    model: nn.Module = amoebanetd(
        num_classes=num_classes, 
        num_layers=num_layers, 
        num_filters=num_filters,)

    batch_size, num_microbatches, devices = EXPERIMENTS[experiment]
    checkpoint_strategy = 'except_last' if num_microbatches > 1 else 'always'
    model, devices = pipeline_model(
        model=model, 
        devices=devices, 
        batch_size=batch_size, 
        chunks=num_microbatches, 
        balance=devices, 
        checkpoint=checkpoint_strategy,
        use_self_gpipe=args.use_self_gpipe)

    # ============================================================================
    # Training Related
    optimizer = SGD(model.parameters(), lr=0.1)
    in_device = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    # This experiment cares about only training speed, rather than accuracy.
    # To eliminate any overhead due to data loading, we use fake random 224x224
    # images over 1000 labels.
    dataset_size = 10000
    input = torch.rand(batch_size, 3, 224, 224, device=in_device)
    target = torch.randint(num_classes, (batch_size,), device=out_device)
    data = [(input, target)] * (dataset_size//batch_size)

    if dataset_size % batch_size != 0:
        last_input = input[:dataset_size % batch_size]
        last_target = target[:dataset_size % batch_size]
        data.append((last_input, last_target))

    # TRAIN =======================================================================================
    hr()
    print_log("Profiling...")
    model.train()
    torch.cuda.empty_cache()

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        torch.cuda.synchronize(in_device)
        start_time = time.time()

        data_trained = 0
        for i, (input, target) in enumerate(data):
            data_trained += input.size(0)

            output = model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize(in_device)

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = time.time() - start_time
        throughput = dataset_size / elapsed_time

        print_log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
            '' % (epoch+1, epochs, throughput, elapsed_time), clear=True)
        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    hr()
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)
        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)
    hr()

    # RESULT ======================================================================================
    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n
    print_log('| %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (throughput, elapsed_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS.keys())
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--skip-epochs', type=int, default=0)
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('--use-self-gpipe', action='store_true')
    args = parser.parse_args()

    main(args)
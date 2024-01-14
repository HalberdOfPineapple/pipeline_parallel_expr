import os
import sys 
import random
import numpy as np
import argparse
from time import perf_counter

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from typing import List, Tuple, Dict, Any, Union
from collections import OrderedDict

from torchgpipe import GPipe as tGPipe
from MyGPipe import GPipe as SelfGPipe
from CleanParallel.parallel import GPipe as CleanGPipe
from resnet import build_resnet, resnet50, resnet100
from gpt_2 import build_GPT2, gpt2_small

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

curr_dir = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(curr_dir, 'data')
CKPT_DIR = os.path.join(curr_dir, 'ckpt')
TRACE_DIR = os.path.join(curr_dir, 'trace')

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hr():
    print('-' * 80)

PARTITION_PLANS = {
    'resnet50': [7, 4, 6, 6], # 23 layers, 4 + ([3, 4, 6, 3]) + 3
    'resnet100': [14, 14, 14, 14], # 56 layers, 4 + ([3, 13, 30, 3]) + 3
    'gpt2_small': [5, 3, 3, 4], # 15 layers, 2 + 12 + 1
}

def pipeline_model(
        model: torch.nn.Module, 
        model_type: str, 
        gpipe_type:str='no_pipe', 
        num_micro_batches: int=4,
        checkpoint_strategy: str='except_last',
    ) -> Tuple[torch.nn.Module, List[int], List[torch.device]]:
    
    if gpipe_type == 'no_pipe':
        print("No pipeline parallelism is used...")
        model = model.cuda() if torch.cuda.is_available() else model
        return model, None, [torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')]
    
    partition_plan = PARTITION_PLANS[model_type]
    if gpipe_type == 'self':
        print("Using Self-implemented GPipe...")
        model = SelfGPipe(
            module=model, 
            partition_plan=partition_plan,
            num_micro_batches=num_micro_batches,
            checkpoint_strategy=checkpoint_strategy,
        )
    elif gpipe_type == 'torchgpipe':
        print("Using torchgpipe...")
        model = tGPipe(
            model, 
            chunks=num_micro_batches, 
            balance=partition_plan,
            checkpoint=checkpoint_strategy,
        )
    elif gpipe_type == 'clean':
        print("Using CleanParallel...")
        model = CleanGPipe(
            model, 
            chunks=num_micro_batches, 
            balance=partition_plan,
            checkpoint=checkpoint_strategy,
        )
    else:
        raise ValueError(f'Invalid gpipe_type: {gpipe_type}')

    return model, partition_plan, model.devices

def build_train_stuffs(model: nn.Module, model_type: str, batch_size: int, data_dir:str= DATA_DIR):
    if model_type == 'resnet50' or model_type == 'resnet100':
        from resnet import build_train_stuffs as resnet_train_stuffs
        return resnet_train_stuffs(model, batch_size, data_dir)
    elif model_type == 'gpt2_small':
        from gpt_2 import build_train_stuffs as gpt2_train_stuffs
        return gpt2_train_stuffs(model, batch_size, data_dir)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')



def train(
        ckpt_save_path: str, 
        gpipe_type:str=None, 
        model_type: str='resnet50', 
        data_dir: str = DATA_DIR,
        num_micro_batches: int=4,
        checkpoint_strategy: str='except_last',
    ):
    """Profiling experiment"""

    # ============================================================================
    hr()
    print(f'Experiment on {model_type} with {gpipe_type} GPipe')
    num_epoches = 1
    batch_size = 128

    # ============================================================================
    # Model Related
    hr()
    print("Setting up model...")
    if model_type == 'resnet50':
        model = resnet50(num_classes=10)
    elif model_type == 'resnet100':
        model = resnet100(num_classes=10)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

    train_loader, criterion, optimizer = build_train_stuffs(model, model_type, batch_size)
    model, partition_plan, devices = pipeline_model(model, 
                        gpipe_type=gpipe_type, 
                        model_type=model_type, 
                        num_micro_batches=num_micro_batches,
                        checkpoint_strategy=checkpoint_strategy,)
    input_device, ouptut_device = devices[0], devices[-1]

    # ==================================================================================
    # Model Parameters
    hr()
    print("Setting up parameters...")
    param_count = sum(p.storage().size() for p in model.parameters())
    param_size = sum(p.storage().size() * p.storage().element_size() for p in model.parameters())
    param_scale = 3  # param + grad + SGD.momentum

    print(f'# of Model Parameters: {param_count:,}')
    print(f'Total Model Parameter Memory: {param_size * param_scale:,} Bytes ({param_size * param_scale / 1024**2:.2f} MiB)')

    # ============================================================================
    # Profiling by training
    hr()
    print("Profiling...")
    
    # Prepare statistics for profiling
    model.train()
    torch.cuda.empty_cache()
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    forward_times, backward_times = [], []

    for ep in range(1, num_epoches + 1):
        train_loss = correct = total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(input_device), targets.to(ouptut_device)

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

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print("Step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                    idx + 1, len(train_loader), ep, num_epoches,
                    train_loss / (idx + 1), 100.0 * correct / total,))
        torch.save(model.state_dict(), ckpt_save_path)
        

    # Save memory profiling statistics
    hr()
    max_memory = 0
    for d in devices:
        torch.cuda.synchronize(d)
        max_memory += torch.cuda.max_memory_reserved(d)
    latent_size = max_memory - param_size * param_scale

    print(f'Peak Activation Memory: {latent_size:,} Bytes ({latent_size / 1024**2:.2f} MiB)')
    print(f'Total Memory: {max_memory:,} Bytes ({max_memory / 1024**2:.2f} MiB)')
    print(f'Average Forward Time: {sum(forward_times)/len(forward_times):.3f} sec')
    print(f'Average Backward Time: {sum(backward_times)/len(backward_times):.3f} sec')
    print("Profiling Completed.")

    # ============================================================================
    # Max memory per device
    hr()
    print("Max memory per device:")
    for d in devices:
        memory_usage = torch.cuda.memory_reserved(d)
        print(f'{d!s}: {memory_usage:,} Bytes ({memory_usage / 1024**2:.2f} MiB)')

def eval(ckpt_save_path, gpipe_type:bool=False, model_type:str= 'resnet50', data_dir: str= DATA_DIR):
    print('-' * 80)
    print('Evaluation Start...')

    batch_size = 128
    if model_type == 'resnet50':
        model = resnet50(num_classes=10)
    elif model_type == 'resnet100':
        model = resnet100(num_classes=10)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')
    
    model, partition_plan, devices = pipeline_model(model, model_type=model_type, gpipe_type=gpipe_type)
    model.load_state_dict(torch.load(ckpt_save_path))
    input_device, ouptut_device = devices[0], devices[-1]

    evalset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    eval_loader = DataLoader(
        evalset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    result = {}
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(input_device), targets.to(ouptut_device)
            outputs = model(inputs)

            num_right = torch.sum(torch.argmax(outputs, dim=-1) == targets).item()
            result['right'] = result.get('right', 0) + num_right
            result['wrong'] = result.get('wrong', 0) + (batch_size - num_right)

    correct = result['right']
    total = result['right'] + result['wrong']
    print(f"Accuracy = {correct / total:.3f} ({correct} / {total})")                                      

if __name__ == "__main__":
    init_seed(999)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpipe_type", "-g", default='no_pipe', type=str, help="Type of GPipe")
    argparser.add_argument("--model_type", "-m", default='resnet50', type=str, help="Type of Model")
    argparser.add_argument("--num_micro", "-n", default=4, type=int, help="Number of micro batches")
    argparser.add_argument("--checkpoint_strategy", "-c", default='except_last', type=str, help="Checkpoint strategy")
    args = argparser.parse_args()

    gpipe_type = args.gpipe_type
    model_type = args.model_type
    num_micro_batches = args.num_micro
    save_checkpoint = os.path.join(CKPT_DIR, f'{model_type}_{gpipe_type}_{num_micro_batches}.bin')

    train(
        save_checkpoint, 
        gpipe_type=gpipe_type, 
        model_type=model_type,
        num_micro_batches=num_micro_batches,
        checkpoint_strategy=args.checkpoint_strategy,
    )
    # eval(save_checkpoint, gpipe_type=gpipe_type)
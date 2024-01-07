import os
import sys 
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from torchgpipe import GPipe as tGPipe
from GPipe import GPipe as SelfGPipe

curr_dir = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(curr_dir, 'data')
CKPT_DIR = os.path.join(curr_dir, 'ckpt')
TRACE_DIR = os.path.join(curr_dir, 'trace')
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_model():
    linear1 = torch.nn.Linear(28 * 28, 28)
    relu = torch.nn.ReLU()
    linear2 = torch.nn.Linear(28, 10)
    return torch.nn.Sequential(linear1, relu, linear2)

def pipeline_model(model: torch.nn.Module, gpipe_type:str=None):
    if not gpipe_type:
        print("No pipeline parallelism is used...")
        model = model.cuda() if torch.cuda.is_available() else model
    elif gpipe_type == 'self':
        print("Using Self-implemented GPipe...")
        model = SelfGPipe(
            module=model, partition_plan=[1, 1, 1],
            num_micro_batches=4,
            devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
            checkpoint_strategy='except_last',
        )
    elif gpipe_type == 'torchgpipe':
        print("Using torchgpipe...")
        model = tGPipe(model, chunks=4, balance=[1, 1, 1],
              devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
              checkpoint='except_last',)
    else:
        raise ValueError(f'Invalid gpipe_type: {gpipe_type}')
    return model

def train(ckpt_save_path, gpipe_type:str=None, data_dir: str = DATA_DIR):
    print('-' * 80)
    print('Training Start...')
    model = build_model()
    model.train()

    model = pipeline_model(model, gpipe_type=gpipe_type)

    optimzier = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 128
    mnist = MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    steps, num_epoches = 0, 1
    print("Profiling...")
    trace_dir = os.path.join(TRACE_DIR, gpipe_type)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(num_epoches):
            for batch in train_loader:
                prof.step()

                steps += 1
                input, target = (batch[0].cuda(), batch[1].cuda()) if torch.cuda.is_available() else (batch[0], batch[1])
                input = input.view([-1, 28 * 28]) # (batch_size, 28 * 28)

                output = model(input)

                if hasattr(model, 'devices'):
                    target = target.to(model.devices[-1])
                loss = criterion(output, target)

                optimzier.zero_grad()
                loss.backward()
                optimzier.step()

            print(f'Epoch: {i}, step: {steps}, loss: {loss.data}')
            torch.save(model.state_dict(), ckpt_save_path)

def eval(ckpt_save_path, gpipe_type:bool=False, data_dir: str= DATA_DIR):
    print('-' * 80)
    print('Evaluation Start...')
    model = build_model()
    model.eval()
    
    model = pipeline_model(model, gpipe_type=gpipe_type)
    model.load_state_dict(torch.load(ckpt_save_path))

    mnist = MNIST(root=data_dir, train=False, download=False, transform=transforms.ToTensor())
    eval_loader = DataLoader(mnist, batch_size=1, shuffle=True)

    result = {}
    for i, batch in enumerate(eval_loader):
        input = batch[0].view([-1, 28 * 28]) # (1, 784)
        input = input.cuda() if torch.cuda.is_available() else input

        output = model(input)

        if torch.argmax(output, dim=-1).item() == batch[1].item():
            result['right'] = result.get('right', 0) + 1
        else:
            result['wrong'] = result.get('wrong', 0) + 1
    
    print(f"Accuracy = {result['right'] / len(eval_loader)} ({result['right']} / {len(eval_loader)})")                                      

if __name__ == "__main__":
    init_seed(999)
    save_checkpoint = os.path.join(CKPT_DIR, 'mnist_model.bin')
    gpipe_type = sys.argv[1] if sys.argv[1] else None

    train(save_checkpoint, gpipe_type=gpipe_type)
    eval(save_checkpoint, gpipe_type=gpipe_type)
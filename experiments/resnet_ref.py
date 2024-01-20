import os
import sys 
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from torchgpipe import GPipe as tGPipe
from MyGPipe import GPipe as SelfGPipe

curr_dir = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(curr_dir, 'data')
CKPT_DIR = os.path.join(curr_dir, 'ckpt')
TRACE_DIR = os.path.join(curr_dir, 'trace')
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pipeline_model(model: torch.nn.Module, gpipe_type:str=None):
    # partition_plan = [len]
    if not gpipe_type:
        print("No pipeline parallelism is used...")
        model = model.cuda() if torch.cuda.is_available() else model
    elif gpipe_type == 'self':
        print("Using Self-implemented GPipe...")
        model = SelfGPipe(
            module=model, partition_plan=[1, 1, 1], num_micro_batches=4,
            devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
            checkpoint_strategy= 'except_last',
        )
    elif gpipe_type == 'torchgpipe':
        print("Using torchgpipe...")
        model = tGPipe(model, chunks=4, balance=[1, 1, 1],
              devices=['cuda:0', 'cuda:1', 'cuda:2'] if torch.cuda.is_available() else ['cpu', 'cpu', 'cpu'],
              checkpoint='except_last',)
    else:
        raise ValueError(f'Invalid gpipe_type: {gpipe_type}')
    return model

def build_train_stuffs(model: nn.Module, batch_size: int):
    # 1. define network
    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
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
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    return train_loader, criterion, optimizer



def train(ckpt_save_path, gpipe_type:str=None, data_dir: str = DATA_DIR):
    print('-' * 80)
    print('Training Start...')
    num_epoches = 1
    batch_size = 128

    model = torchvision.models.resnet18(num_classes=10)
    train_loader, criterion, optimizer = build_train_stuffs(model, batch_size)
    model = pipeline_model(model, gpipe_type=gpipe_type)
    
    print("Profiling...")
    model.train()
    # trace_dir = os.path.join(TRACE_DIR, gpipe_type)
    for ep in range(1, num_epoches + 1):
        train_loss = correct = total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        num_epoches,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
        torch.save(model.state_dict(), ckpt_save_path)

def eval(ckpt_save_path, gpipe_type:bool=False, data_dir: str= DATA_DIR):
    print('-' * 80)
    print('Evaluation Start...')
    model = torchvision.models.resnet18(num_classes=10)
    model.eval()
    
    model = pipeline_model(model, gpipe_type=gpipe_type)
    model.load_state_dict(torch.load(ckpt_save_path))

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
        evalset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )

    result = {}
    for idx, (inputs, targets) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        if torch.argmax(outputs, dim=-1).item() == targets.item():
            result['right'] = result.get('right', 0) + 1
        else:
            result['wrong'] = result.get('wrong', 0) + 1
    
    print(f"Accuracy = {result['right'] / len(eval_loader)} ({result['right']} / {len(eval_loader)})")                                      

if __name__ == "__main__":
    init_seed(999)
    
    gpipe_type = sys.argv[1] if sys.argv[1] else None

    train(save_checkpoint, gpipe_type=gpipe_type)
    eval(save_checkpoint, gpipe_type=gpipe_type)
import torch
from GPipe import GPipe

def checkpoint_multi_stream_v1():
    print('=' * 50)
    print('Basic experiment with GPipe')

    torch.manual_seed(999)
    a = torch.tensor(2.0, requires_grad=True, device='cuda:0')
    b = torch.tensor(4.0, requires_grad=True, device='cuda:0')

    
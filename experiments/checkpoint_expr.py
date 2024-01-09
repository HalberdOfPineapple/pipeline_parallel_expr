import torch
import MyGPipe.stream_utils as stream_utils
from MyGPipe.checkpointing import CheckPointingV0, CheckPointingCLS, Join

# --------------------------------------------------------------------
def func_1(a: torch.Tensor, b: torch.Tensor):
    return a * a + b * b

def func_2(c: torch.Tensor):
    return 2 * c

# --------------------------------------------------------------------
def checkpoint_single_stream():
    print('=' * 50)
    print('Experiment: checkpointing within a single CUDA stream')
    torch.manual_seed(999)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(4.0, requires_grad=True)

    c: torch.Tensor = CheckPointingV0.apply(func_1, a, b)
    c.backward()
    
    print(c) 
    print(a.grad) 
    print(b.grad)
    print()

def checkpoint_multi_stream_naive():
    # Called 'naive' because the output is not really done in stream_1 (device cuda:0 directed)
    # We need further copying mechanism to transfer c to the second device's stream 
    # and backward the gradients between these two streams

    print('=' * 50)
    print('Experiment: checkpointing within multiple CUDA streams (naive)')
    torch.manual_seed(999)
    a = torch.tensor(2.0, requires_grad=True, device='cuda:0')
    b = torch.tensor(4.0, requires_grad=True, device='cuda:0')

    stream_0 = torch.cuda.default_stream('cuda:0')
    stream_1 = torch.cuda.default_stream('cuda:1')

    with torch.cuda.stream(stream_0):
        c = CheckPointingV0.apply(func_1, a, b)
    with torch.cuda.stream(stream_1):
        output = CheckPointingV0.apply(func_2, c)
    output.backward()

    print(output) 
    print(a.grad) 
    print(b.grad)
    print()

def checkpoint_multi_stream():
    print('=' * 50)
    print('Experiment: checkpointing within multiple CUDA streams')

    torch.manual_seed(999)
    a = torch.tensor(2.0, requires_grad=True, device='cuda:0')
    b = torch.tensor(4.0, requires_grad=True, device='cuda:0')

    stream_0_copy = torch.cuda.Stream('cuda:0')
    stream_0_cal = torch.cuda.default_stream('cuda:0')

    stream_1_copy = torch.cuda.Stream('cuda:1')
    stream_1_cal = torch.cuda.default_stream('cuda:1')

    with torch.cuda.stream(stream_0_cal):
        c = CheckPointingV0.apply(func_1, a, b)

    stream_utils.stream_wait(curr_stream=stream_0_copy, prev_stream=stream_0_cal, data=c)
    c = stream_utils.stream_copy(stream_0_copy, stream_1_copy, c)
    stream_utils.stream_wait(curr_stream=stream_1_cal, prev_stream=stream_1_copy, data=c)

    with torch.cuda.stream(stream_1_cal):
        output = CheckPointingV0.apply(func_2, c)
    output.backward()

    print("output: ", output)
    print("a.grad: ", a.grad)
    print("b.grad: ", b.grad) 
    print()


def checkpoint_multi_stream_v1():
    print('=' * 50)
    print('Experiment: checkpointing (V1) within multiple CUDA streams')

    torch.manual_seed(999)
    a = torch.tensor(2.0, requires_grad=True, device='cuda:0')
    b = torch.tensor(4.0, requires_grad=True, device='cuda:0')

    stream_0_copy = torch.cuda.Stream('cuda:0')
    stream_0_cal = torch.cuda.default_stream('cuda:0')

    stream_1_copy = torch.cuda.Stream('cuda:1')
    stream_1_cal = torch.cuda.default_stream('cuda:1')

    # --------------------------------------------------------------------
    # Normal forward pass (.checkpoint) + extra aux tensor (add compute node for backward pass) + wait + copy
    checkpoint = CheckPointingCLS(stream_0_cal, func_1, a, b)
    c = checkpoint.checkpoint()
    aux_tensor = checkpoint.recompute(c)
    c = Join.apply(c, aux_tensor)

    c = stream_utils.stream_wait(stream_0_copy, stream_0_cal, c)
    c = stream_utils.stream_copy(stream_0_copy, stream_1_copy, c)

    # --------------------------------------------------------------------
    c = stream_utils.stream_wait(stream_1_cal, stream_1_copy, c)
    checkpoint = CheckPointingCLS(stream_1_cal, func_2, c)
    d = checkpoint.checkpoint()
    aux_tensor = checkpoint.recompute(d)
    output = Join.apply(d, aux_tensor)


    output.backward()
    print("output: ", output)
    print("a.grad: ", a.grad)
    print("b.grad: ", b.grad) 
    print()


if __name__ == '__main__':
    # checkpoint_single_stream()
    # checkpoint_multi_stream_naive()
    # checkpoint_multi_stream()
    checkpoint_multi_stream_v1()
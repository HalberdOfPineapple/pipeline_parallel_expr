import torch
from contextlib import contextmanager
from .microbatch import Batch

class CPUStream():
    device = 'cpu'
cpu_stream = CPUStream()

def str_to_device(device):
    return torch.device(device) if isinstance(device, str) else device

def init_stream(device):
    device = str_to_device(device)
    if device.type != 'cuda':
        return cpu_stream
    
    return torch.cuda.Stream(device)

def get_default_stream(device):
    device = str_to_device(device)
    if device.type != 'cuda':
        return cpu_stream

    return torch.cuda.default_stream(device)

def get_current_stream(device):
    device = str_to_device(device)
    if device.type != 'cuda':
        return cpu_stream

    return torch.cuda.current_stream(device)

@contextmanager
def use_stream(stream):
    if isinstance(stream, CPUStream):
        yield
        return
    
    with torch.cuda.stream(stream):
        yield

def record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream):
    if isinstance(stream, CPUStream):
        return
    
    tensor = tensor.new_empty([0]).set_(tensor.untyped_storage())
    tensor.record_stream(stream)

def wait_stream(source_stream, target_stream):
    if isinstance(target_stream, CPUStream):
        return
    
    if isinstance(source_stream, CPUStream):
        target_stream.synchronize()
    else:
        # https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream
        source_stream.wait_stream(target_stream)


class StreamCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, curr_copy_stream, next_copy_stream, *tensors):
        # Note curr_copy_stream and next_copy_stream are specifically responsible for data copying
        # next_cal_stream should be the calculation stream on the target device

        ctx.curr_copy_stream = curr_copy_stream
        ctx.next_copy_stream = next_copy_stream

        outputs = []
        next_cal_stream = get_current_stream(next_copy_stream.device)

        with use_stream(curr_copy_stream), use_stream(next_copy_stream):
            for tensor in tensors:
                tensor_copied = tensor.to(next_copy_stream.device)
                outputs.append(tensor_copied)

                # Preventing the data being released before next calculation stream finishes
                record_stream(tensor, next_copy_stream)
                record_stream(tensor_copied, next_cal_stream)
        
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        curr_copy_stream = ctx.curr_copy_stream
        next_copy_stream = ctx.next_copy_stream

        grad_outputs = []
        curr_cal_stream = get_current_stream(curr_copy_stream.device)

        with use_stream(curr_copy_stream), use_stream(next_copy_stream):
            for grad in grads:
                grad_copied = grad.to(curr_copy_stream.device)
                grad_outputs.append(grad_copied)

                record_stream(grad, next_copy_stream)
                record_stream(grad_copied, curr_cal_stream)

        return (None, None) + tuple(grad_outputs)
        
# def stream_copy(curr_copy_stream: torch.cuda.Stream, next_copy_stream: torch.cuda.Stream, data):
#     if isinstance(data, (list, tuple)):
#         # outputs will be a tuple as well
#         return StreamCopy.apply(curr_copy_stream, next_copy_stream, *data)
#     else:
#         # outputs should be a single tensor
#         return StreamCopy.apply(curr_copy_stream, next_copy_stream, data)[0]
    
def stream_copy(curr_copy_stream: torch.cuda.Stream, next_copy_stream: torch.cuda.Stream, batch: Batch):
    batch[:] = StreamCopy.apply(curr_copy_stream, next_copy_stream, *batch)
    return batch
    

class Wait(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, curr_stream, prev_stream, *tensors):
        ctx.prev_stream = prev_stream
        ctx.curr_stream = curr_stream

        wait_stream(curr_stream, prev_stream)

        return tuple(tensor.detach() for tensor in tensors)

    @staticmethod
    def backward(ctx, *tensors):
        prev_stream, curr_stream = ctx.prev_stream, ctx.curr_stream

        wait_stream(prev_stream, curr_stream)
        return (None, None) + tensors

# def stream_wait(curr_stream, prev_stream, data):
#     if isinstance(data, (list, tuple)):
#         return Wait.apply(curr_stream, prev_stream, *data)
#     else:
#         return Wait.apply(curr_stream, prev_stream, data)[0]

def stream_wait(curr_stream, prev_stream, batch: Batch) -> None:
    batch[:] = Wait.apply(curr_stream, prev_stream, *batch)
    return batch
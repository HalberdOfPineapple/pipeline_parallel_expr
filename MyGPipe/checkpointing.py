import torch
from typing import Callable

# ------------------- Checkpointing Base Version ------------------------
class CheckPointingV0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, *args):
        ctx.func = func
        ctx.save_for_backward(*args) # saves all the input tensors

        # Saves rng_state for restoring random settings when doing recomputation in backward pass
        ctx.cpu_rng_state = torch.get_rng_state() # Returns the random number generator state as a torch.ByteTensor
        ctx.gpu_rng_state = torch.cuda.get_rng_state(args[0].device) if args[0].is_cuda else None

        # Do forward pass without gradient tracking
        # Note that even without gradient tracking here, the input tensors in args can be set with requires_grad=True for later recomputation
        with torch.no_grad():
            output = func(*args)

        return output
    
    @staticmethod
    def backward(ctx, *grads):
        args = ctx.saved_tensors # saved input tensors

        # Restore rng_state for restoring random settings when doing recomputation in backward pass
        with torch.random.fork_rng(devices=[args[0].device] if args[0].is_cuda else None):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.gpu_rng_state is not None:
                torch.cuda.set_rng_state(ctx.gpu_rng_state)
        
            # Recompute forward pass with gradient tracking
            tensor_require_grads = [tensor.detach().requires_grad_(tensor.requires_grad) for tensor in args]
            with torch.enable_grad():
                forward_outputs = ctx.func(*tensor_require_grads)
        
        # Compute gradients
        torch.autograd.backward(forward_outputs, grad_tensors=grads)

        # Return. 
        # Note that the first element of the return tuple is None because the first argument of forward() is func
        return (None, ) + tuple([tensor.grad for tensor in tensor_require_grads])


# ------------------- Checkpointing Version 1 ------------------------
class CheckPointingCLS():
    def __init__(self, stream, func, *tensors):
        self.stream: torch.cuda.Stream = stream
        self.func: Callable = func
        self.tensors = tensors

        #  Save shared parameters for checkpointing and recomputation, including rng_state, computation graphs etc.
        self.shared_parameters = {}
    
    def checkpoint(self):
        with torch.cuda.stream(self.stream):
            aux_tensor = torch.tensor(0.0, device=self.tensors[0].device, requires_grad=True)
            return CheckPointingV1.apply(self.func, self.shared_parameters, aux_tensor, *self.tensors)
    
    def recompute(self, data):
        with torch.cuda.stream(self.stream):
            aux_tensor = RecomputeV1.apply(self.func, self.shared_parameters, data, *self.tensors)
            return aux_tensor


class CheckPointingV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, shared_parameters, aux_tensor, *tensors):
        ctx.func = func
        ctx.shared_parameters = shared_parameters
        ctx.save_for_backward(*tensors)

        # Saves rng_state for restoring random settings when doing recomputation in backward pass
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.gpu_rng_state = torch.cuda.get_rng_state(tensors[0].device) if tensors[0].is_cuda else None
        ctx.shared_parameters['rng_state'] = (ctx.cpu_rng_state, ctx.gpu_rng_state)

        with torch.no_grad():
            output = func(*tensors)
        return output
    
    @staticmethod
    def backward(ctx, *grads):
        recomputed_outputs, tensors_require_grads = ctx.shared_parameters['recomputed']
        torch.autograd.backward(recomputed_outputs, grad_tensors=grads)
        return (None, None, None, ) + tuple([tensor.grad for tensor in tensors_require_grads])


class RecomputeV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, shared_parameters, data, *tensors):
        ctx.func = func
        ctx.shared_parameters = shared_parameters
        ctx.save_for_backward(*tensors)

        aux_tensor = torch.tensor(0.0, device=tensors[0].device)
        return aux_tensor

    @staticmethod
    def backward(ctx, *grads):
        """Specifically responsible for the recomputation during backward"""
        # tensors saved by ctx.save_for_backward(*tensors) in forward pass
        saved_tensors = ctx.saved_tensors

        with torch.random.fork_rng(devices=[saved_tensors[0].device] if saved_tensors[0].is_cuda else None):
            cpu_rng_state, gpu_rng_state = ctx.shared_parameters['rng_state']
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_state is not None:
                torch.cuda.set_rng_state(gpu_rng_state)
            
            tensors_require_grads = [tensor.detach().requires_grad_(tensor.requires_grad) for tensor in saved_tensors]
            with torch.enable_grad():
                recomputed_outputs = ctx.func(*tensors_require_grads)
        
        ctx.shared_parameters['recomputed'] = (recomputed_outputs, tensors_require_grads)

        # Recompute only needs to perform the recomputation to build the computation graph
        # but it does not need to have any gradients itself
        return (None, None, None,) + tuple([None for _ in tensors_require_grads])


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, aux_tensor):
        return input_tensor.detach()
    
    @staticmethod
    def backward(ctx, grads):
        return grads, None
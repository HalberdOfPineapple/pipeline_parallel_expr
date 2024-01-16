import torch
from typing import Callable, Tuple, Dict

from .stream_utils  import use_stream, get_default_stream
from .microbatch import Batch

aux_tensor_map: Dict[Tuple[torch.device, bool], torch.Tensor] = {}
def get_aux_tensor(device: torch.device, *, requires_grad: bool) -> torch.Tensor:
    key = (device, requires_grad)
    try:
        aux_tensor = aux_tensor_map[key]
    except KeyError:
        with use_stream(get_default_stream(device)):
            aux_tensor = torch.empty(0, device=device, requires_grad=requires_grad)
        aux_tensor_map[key] = aux_tensor

    return aux_tensor


class JoinV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Join', input: torch.Tensor, aux_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore
        return input.detach()

    @staticmethod
    def backward(ctx: 'Join', grad_input: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore
        return grad_input, None

def join(input: torch.Tensor, aux_tensor: torch.Tensor) -> torch.Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or aux_tensor.requires_grad):
        input = JoinV2.apply(input, aux_tensor)

    return input


class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Fork', input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        aux_tensor = get_aux_tensor(input.device, requires_grad=False)
        return input.detach(), aux_tensor.detach()

    @staticmethod
    def backward(ctx: 'Fork', grad_input: torch.Tensor, grad_grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        return grad_input

def fork(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, aux_tensor = Fork.apply(input)
    else:
        aux_tensor = get_aux_tensor(input.device, requires_grad=False)

    return input, aux_tensor

def depend(fork_batch: Batch, join_batch: Batch) -> None:
    fork_batch, aux_tensor = fork(fork_batch[0])
    join_batch[0] = join(join_batch[0], aux_tensor)
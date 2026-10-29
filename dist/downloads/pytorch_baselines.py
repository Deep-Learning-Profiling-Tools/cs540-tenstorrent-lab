"""PyTorch references for the logical exercise inputs and outputs."""
import torch


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a, b: float32 [N]; result: float32 [N]
    return a + b


def ragged_sum_reference(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # x: float32 [R, L]; lengths: int64 [R]; result: float32 [R]
    columns = torch.arange(x.shape[1], device=x.device)
    valid = columns[None, :] < lengths[:, None]
    values = x.masked_fill(~valid, 0.0)
    return values.sum(dim=1)


def ragged_softmax_reference(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    # x: float32 [R, L]; lengths: int64 [R]; result: float32 [R, L]
    columns = torch.arange(x.shape[1], device=x.device)
    valid = columns[None, :] < lengths[:, None]
    logits = x.masked_fill(~valid, -torch.inf)
    # Give empty rows finite logits before softmax; zero them below.
    logits = torch.where(lengths[:, None] > 0, logits, 0.0)
    result = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
    return result

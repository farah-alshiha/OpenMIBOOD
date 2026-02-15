import torch
import torch.nn.functional as F

def pixel_1_minus_maxsoftmax(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=0)
    m, _ = p.max(dim=0)
    return 1.0 - m

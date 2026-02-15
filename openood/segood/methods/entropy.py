import torch
import torch.nn.functional as F

def pixel_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # logits: [C, H, W]
    p = F.softmax(logits, dim=0)
    ent = -(p * (p + eps).log()).sum(dim=0)
    return ent

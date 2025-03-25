import torch
import torch.nn.functional as F

def sample_gumbel(shape, device, dtype, eps=1e-6):
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=3.0):
    y = logits + sample_gumbel(logits.size(), logits.device, logits.dtype)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=3.0, hard=True):
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = F.one_hot(ind, num_classes=shape[-1]).to(dtype=logits.dtype)
    y_hard = (y_hard - y).detach() + y
    return y_hard

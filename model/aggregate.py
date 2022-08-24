import torch
import torch.nn.functional as F


# Soft aggregation from STM
def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=dim, keepdim=True),
        prob
    ], dim).clamp(1e-7, 1-1e-7) # B x (max_obj_num+1) x H x W
    logits = torch.log((new_prob /(1-new_prob)))
    prob = F.softmax(logits, dim=dim)

    if return_logits:
        return logits, prob
    else:
        return prob
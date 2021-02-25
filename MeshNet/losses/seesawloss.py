import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from typing import Union


class NormalizedLinear(nn.Module):
    r"""Normalized linear layer which was proposed by Jiaqi Wang et. al. in 
        their technical report for LVIS workshop at ECCV 2020.
        https://arxiv.org/pdf/2008.10032.pdf
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        tau: a temperature factor, set as 20 by default by following the paper.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, tau: float = 20) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.tau = tau

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return F.linear(
            input / torch.norm(input, dim=-1, keepdim=True),
            self.tau * self.weight /
            torch.norm(self.weight, dim=-1, keepdim=True),
            self.bias
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

    
class DistibutionAgnosticSeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 for default following the paper.
    """

    def __init__(self, p: float = 0.8):
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.s = None
        self.class_counts = None

    def forward(self, logits, targets):
        if self.class_counts is None:
            self.class_counts = targets.sum(axis=0) + 1 # to prevent devided by zero.
        else:
            self.class_counts += targets.sum(axis=0)

        conditions = self.class_counts[:, None] > self.class_counts[None, :]
        trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
        falses = torch.ones(len(self.class_counts), len(self.class_counts))
        self.s = torch.where(conditions, trues, falses)

        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()



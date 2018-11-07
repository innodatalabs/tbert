import torch
import math


def gelu(x):
    '''Gaussian Error Linear Unit - a smooth version of RELU'''
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf

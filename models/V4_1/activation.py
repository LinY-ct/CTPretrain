#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import math
from torch import nn
from torch.nn import functional as F
Tensor = torch.Tensor
e = math.e
from torch.nn.parameter import Parameter
import cmath

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class lush(nn.Module):
    def __init__(self):
        super(lush, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, x * torch.tanh(F.softplus(x)), x)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))
    
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class lush2(nn.Module):
    def __init__(self):
        super(lush2, self).__init__()
    def forward(self, x):
        return torch.where(x<0, -(-x+1)**((1/(-x+1)))+1,x)

class lush_2(nn.Module):
    def __init__(self):
        super(lush_2, self).__init__()

    def forward(self, x):
        return torch.where(x<0,(x * (e ** (x+1) - e ** (-(x+1))) / (e **(x+1) + e ** (-(x+1)))),x)

class lush_1655(nn.Module):
    def __init__(self):
        super(lush_1655, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, 1.655*(x * torch.tanh(F.softplus(x))), x)

        
class tanh(nn.Module):
    def __init__(self):
        super(tanh, self).__init__()
    def forword(self, x):
        return (e ** x - e ** (-x)) / (e ** x + e ** (-x))


'''
class lush(nn.Module):
    def __init__(self):
        super(lush, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, math.log((math.exp(x) + 1 )) * x * torch.tanh(F.softplus(x)), x)
'''
class exlu(nn.Module):
    def __init__(self):
        super(exlu, self).__init__()
    def forward(self, x):
        return torch.where(x < 0, -math.exp(0.1*x), x)


class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class gelu_new(nn.Module):
    def __init__(self):
        super(gelu_new, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.sigmoid(x)

class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid,self).__init__()
    def forward(self,x):
        return 1 / (1 + torch.exp(x)) * torch.exp(x)

class leaky_relu(nn.Module):
    def __init__(self):
        super(leaky_relu,self).__init__()
    def forward(self,x):
        return torch.where(x < 0, 0.1 * x, x) 


class relu(nn.Module):
    def __init__(self):
        super(relu,self).__init__()
    def forward(self,x):
        return torch.where(x < 0, 0, x)

class Lms(nn.Module):
    def __init__(self,a,x):
        super(Lms, self).__init__()
        self.a = a
        self.x = x
    def forward(self):
        if 0 < self.x <= self.a:
            return -((1/self.a) ** 2 * self.x - self.a)
        elif (-self.a <= self.x < 0):
            return -((1/self.a) ** 2 *self.x + self.a)
        else:
            return 0
class Lms2(nn.Module):
    def __init__(self,x):
        super(Lms2, self).__init__()
        self.x = x
    def forward(self,x):
        if 0 < x <= 0.5:
            return ((1/0.5) ** 2 * x - 0.5)
        elif (-0.5 <= x < 0):
            return ((1/0.5) ** 2 *x + 0.5)
        else:
            return 0

class guassian_dx(nn.Module):
    def __init__(self,):
        super(guassian_dx,self).__init__()
    def forward(self,x,a_coe,b_coe,c_per):
        return (math.log(a_coe,math.e) * c_per * x * e ** ((-b_coe*x)**2))



ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
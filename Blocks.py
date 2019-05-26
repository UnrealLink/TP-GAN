import torch.nn as nn
from torch.nn.init import kaiming_normal_

def weight_initialize(weight, activation = None):
    
    if hasattr(activation, "negative_slope"):
        kaiming_normal_(weight, a = activation.negative_slope)
        
    else :
        kaiming_normal_(weight, a = 0)

def Sequential(*kargs):
    seq = nn.Sequential(*kargs)
    for layer in reversed(kargs):
        if hasattr( layer , 'out_channels'):
            seq.out_channels = layer.out_channels
            break
        if hasattr( layer , 'out_features'):
            seq.out_channels = layer.out_features
            break
    return seq 

def ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, activation = nn.LeakyReLU(), use_batchnorm = False, init_weight = True) :
    
    blocks = []
    blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if init_weight:
        weight_initialize(blocks[-1].weight, activation)
    
    if use_batchnorm:
        blocks.append(nn.BatchNorm2d(out_channels))
    if not (activation is None):
        blocks.append(activation)  
    
    seq = nn.Sequential(*blocks)
    seq.out_channels = out_channels
    return seq
    

def DeConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, activation = nn.ReLU(), use_batchnorm = False):
    
    blocks = []
    blocks.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
    weight_initialize(blocks[-1].weight, activation)
    
    if use_batchnorm :
        blocks.append(nn.BatchNorm2d(out_channels))    
    if not activation is None:
        blocks.append(activation)
        
    seq = nn.Sequential(*blocks)
    seq.out_channels = out_channels
    return seq

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels = None, kernel_size = 3, stride = 1, padding = 1, activation = nn.LeakyReLU()):
        super(ResidualBlock, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels // stride
        
        self.activation = activation
        self.input = ConvBlock(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, activation = None, use_batchnorm = False)
        
        blocks = []
        padding = (kernel_size - 1)// 2
        blocks.append(ConvBlock(in_channels, in_channels, kernel_size, stride = 1, padding = padding, activation = activation, use_batchnorm = True))
        blocks.append(ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = padding, activation = None, use_batchnorm = True))
        self.block = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x):
        return self.activation(self.block(x) + self.input(x))
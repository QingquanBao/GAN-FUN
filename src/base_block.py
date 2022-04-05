import torch
import torch.nn.functional as F
from torch import nn

def discriminator_block(in_filters, out_filters, bn=True):
    ''' Narrow feature map x2 in terms of width and height
    '''
    block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), 
             nn.LeakyReLU(0.2, inplace=True), 
             nn.Dropout2d(0.25)]
    if bn:
        block.append(nn.BatchNorm2d(out_filters)) #, 0.8))
    return block

def generator_block(in_filters, out_filters, bn=True):
    ''' Scale feature map x2 in terms of width and height
    '''
    block = [nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1)] 
    if bn:
        block.append(nn.BatchNorm2d(out_filters)) #, 0.8))
    block.append(nn.LeakyReLU(0.2, inplace=True))
    return block

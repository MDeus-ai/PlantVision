import torch.nn as nn
from collections import OrderedDict

'''
The Conv2dNormActivation block is the simplest block in this model architecture
It consists of three layers:
    1). The convolutional layer
    2). The batch-normalization layer
    3). An activation function (SiLU) 
'''

class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation_layer=nn.SiLU):
        layers = OrderedDict() # Store the layers
        # Convolution layer
        layers['0'] = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding,
            bias=False
        )
        # BatchNormalization layer
        layers['1'] = nn.BatchNorm2d(out_channels)
        # Activation Layer
        if activation_layer is not None:
            layers['2'] = activation_layer(inplace=True)
        
        super().__init__(layers)


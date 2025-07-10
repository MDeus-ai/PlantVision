import torch.nn as nn
from torchvision.ops import StochasticDepth
from .Conv2dNormActivation import Conv2dNormActivation
from .SE_block import SqueezeExcitation
from collections import OrderedDict

'''
# CONCEPT EXPLANATION OF THE MOBILE INVERTED BOTTLENECK CONVOLUTION (MBConvBlock)
=================================================================================
Summary of what the MBConvBlock does: Expansion -> Depthwise conv -> Squeeze-and-Excitation -> Project -> Residual(Skip) connection

PARAMETERS:
    stride: controls downsampling the spatial resolution
    expand_ratio: Controls the width expansion factor in the bottleneck (6 for EfficientNet-B0 MBConvs)
    se_ratio: Determines how much to shrink inside SE block (normally 0.25)

CONCEPTS:
Expansion gives the nn more room to think before condensing the features again
Depthwise Separable convolutions, there each input channel has its own filter, reducing the number of parameters, hence cheap
'''

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, stochastic_depth_prob=0.0):
        super(MBConv, self).__init__()
        # Add a skip connection if the shape doesn't change across the block
        self.use_res_connection = (stride == 1 and in_channels == out_channels)
        # Calc the number of channels produced during expansion
        hidden_dim = in_channels * expand_ratio

        squeezed_channels = int(in_channels * se_ratio)

        # Build all layers and blocks in the layers ordereddict
        layers = OrderedDict()

        # Expansion layer: A 1x1 conv that expands from in_channels to hidden_dim
        if expand_ratio != 1:
            layers['0'] = Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # Depthwise conv layer: Independent filters(whose # is equal to hidden_dim) for each input channel
        layers[f'{len(layers)}'] = Conv2dNormActivation(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=hidden_dim
        )

        # Squeeze-and-Excitation: Weighs the importance of input feature maps
        layers[f'{len(layers)}'] = SqueezeExcitation(
            in_channels=hidden_dim,
            squeezed_channels=squeezed_channels
        )

        # Projection layer: A 1x1 conv that scales back the channels closing the bottleneck
        layers[f'{len(layers)}'] = Conv2dNormActivation(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=None
        )

        self.block = nn.Sequential(layers) # The complete MBConv block

        # Stochastic depth layer
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connection:
            out = self.stochastic_depth(out)
            return x + out
        else:
            return out

import torch.nn as nn
'''
THE SQUEEZE AND EXCITATION BLOCK
This learns to pay more attention to the most important channels from the input channels 
and puts less attention on irrelevant ones.

HOW IT WORKS
1). It finds the average of each input channel by performing global average pooling
producing a shape (B, C, 1, 1)
2). It then uses them as input into a 2-layered MLP
    Layer-1 performs dim reduction (out_channels=in_channels//reduction) and applies relu activation
    Layer-2 Projects back (out_channels=in_channels) and applies a sigmoid activation outputing channel-wise
    attention scores
3). Finally the attention scores are multiplied by their respective channels (original feature maps)       
'''

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeezed_channels):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=squeezed_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=squeezed_channels, out_channels=in_channels, kernel_size=1)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.scale_activation(self.fc2(scale))
        return x * scale


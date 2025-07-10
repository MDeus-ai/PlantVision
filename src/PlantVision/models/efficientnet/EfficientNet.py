import torch
import torch.nn as nn
import numpy as np
from .Conv2dNormActivation import Conv2dNormActivation
from .MBConv_block import MBConv

scale_values = {
    # (width_mult, depth_mult, resolution, dropout)
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
    'b5': (1.6, 2.2, 456, 0.4),
    'b6': (1.8, 2.6, 528, 0.5),
    'b7': (2.0, 3.1, 600, 0.5),
}
alpha, beta = 1.2, 1.1


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, model_name='b0', stochastic_depth_prob=0.2):
        super(EfficientNet, self).__init__()

        width_mult, depth_mult, resolution, dropout = scale_values[model_name]

        base_config = [
            # [Expand_ratio, in_channels, out_channels, repeats, strides, kernel_size]
            [1, 32, 16, 1, 1, 3],
            [6, 16, 24, 2, 2, 3],
            [6, 24, 40, 2, 2, 5],
            [6, 40, 80, 3, 2, 3],
            [6, 80, 112, 3, 1, 5],
            [6, 112, 192, 4, 2, 5],
            [6, 192, 320, 1, 1, 3],
        ]

        # MBConv Blocks
        self.features = nn.Sequential()

        # Stem: This is the very first block that receives input images
        in_channels = 3
        out_channels = self._make_divisible(32 * width_mult, 8)
        stem = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2, # Downsample to half the resolution
            padding=1
        )
        self.features.add_module('0', stem)
        in_channels = out_channels # For the next stages

        # Calculate the total number of blocks for stochastic depth
        total_blocks = sum(int(np.ceil(repeat*depth_mult)) for _, _, _, repeat, _, _ in base_config)
        block_counter = 0

        # Create stages, each a Sequential module
        for i, (k, c_in_base, c_out, repeat, stride, kernel_size) in enumerate(base_config):
            stage_layers = nn.Sequential()
            out_channels = self._make_divisible(c_out*width_mult, 8)
            num_layers = int(np.ceil(repeat*depth_mult))

            # Loop to assemble blocks for each stage
            for j in range(num_layers):
                s=stride if i==0 else 1 # Only downsample the first block
                # Calc stochastic depth proba for this specific block
                sd_proba = stochastic_depth_prob * float(block_counter) / total_blocks

                block = MBConv(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    kernel_size=kernel_size,
                    stride=s,
                    expand_ratio=k,
                    stochastic_depth_prob=sd_proba
                )
                stage_layers.add_module(f'{j}', block)
                in_channels = out_channels # Update in_channels for the next block
                block_counter += 1
            self.features.add_module(f'{i + 1}', stage_layers)

        # Head
        head_in_channels = in_channels
        head_channels = self._make_divisible(1280*width_mult, 8)
        head = Conv2dNormActivation(
            in_channels=head_in_channels,
            out_channels=head_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.features.add_module(f'{len(base_config) + 1}', head)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=head_channels, out_features=num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return int(new_v)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / np.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
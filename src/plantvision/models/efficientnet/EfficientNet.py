import torch
import torch.nn as nn
from torchvision import models


class EfficientNet(nn.Module):
    """
    A wrapper for the EfficientNet models (b0, b1, b2, b3).
    """
    def __init__(self, num_classes: int, model_name: str = 'b0', pretrained: bool = False, freeze_layers: bool = False):
        super(EfficientNet, self).__init__()

        print(f".")

        # Map our model_name string to the correct torchvision model and weights enum
        model_mapping = {
            'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
        }

        if model_name not in model_mapping:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}. "
                             f"Supported variants are: {list(model_mapping.keys())}")

        model_builder, weights_enum = model_mapping[model_name]

        # Load the model with pre-trained weights or from scratch
        if pretrained:
            print(f"..")
            self.base_model = model_builder(weights=weights_enum)
        else:
            print("...")
            self.base_model = model_builder(weights=None, num_classes=num_classes)

        # Adapt the classifier for a custom number of classes
        if pretrained:
            # Get the number of input features for the classifier layer
            in_features = self.base_model.classifier[1].in_features

            # Replace the final layer with a new one for our number of classes
            self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
            print(f" 🔄️ Replaced classifier head for {num_classes} classes.")

        # Freeze the feature layers (Convolutional stages)
        if freeze_layers:
            print(" ❄️ Freezing feature extraction layers...")
            # Iterate through all parameters in the 'features' part of the model
            for param in self.base_model.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        return self.base_model(x)
import torch
import pytest
from unittest.mock import patch
from plantvision.models.efficientnet.EfficientNet import EfficientNet


def test_efficientnet_instantiation_from_scratch():
    """Tests if the model can be instantiated without pre-trained weights."""
    try:
        model = EfficientNet(num_classes=10, model_name='b0', pretrained=False)
    except Exception as e:
        pytest.fail(f"Instantiation from scratch failed: {e}")


# Patch the `torchvision.models` to avoid actual network downloads during unit tests
@patch('torchvision.models.efficientnet_b0')
def test_efficientnet_instantiation_pretrained(mock_efficientnet_b0):
    """Tests if the pre-training and freezing logic paths are executed."""
    try:
        EfficientNet(
            num_classes=10,
            model_name='b0',
            pretrained=False,
            freeze_layers=True
        )
    except Exception as e:
        pytest.fail(f"Instantiation with pre-training failed: {e}")

    # Assert that the torchvision model builder was called with the 'weights' argument
    mock_efficientnet_b0.assert_called_once()
    assert 'weights' in mock_efficientnet_b0.call_args.kwargs


# The parametrize test for the forward pass
@pytest.mark.parametrize("model_name", ['b0', 'b1'])
def test_efficientnet_forward_pass(model_name):
    num_classes = 10
    batch_size = 2
    img_size = 224
    model = EfficientNet(num_classes=num_classes, model_name=model_name, pretrained=False)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    output = model(dummy_input)
    assert output.shape == (batch_size, num_classes)
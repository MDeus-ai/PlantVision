import torch
import pytest
from PlantVision.models.efficientnet.EfficientNet import EfficientNet


def test_efficientnet_instantiation():
    """
    Tests if the EfficientNet model can be instantiated without errors
    """

    # ARRANGE & ACT
    try:
        model = EfficientNet(model_name='b0', num_classes=10)
    except Exception as e:
        # ASSERT
        pytest.fail(f"EfficientNet instantiation failed with an exception: {e}")



@pytest.mark.parametrize('model_name', ['b0', 'b1', 'b2', 'b3'])
def test_efficientnet_forward_pass(model_name):
    """
    A smoke test to ensure the model can perform a forward pass.
    Tests multiple model variations using parametrize
    """

    # ARRANGE
    num_classes = 10
    batch_size = 4
    img_size = 224

    # A model instance
    model = EfficientNet(model_name=model_name, num_classes=num_classes)
    model.eval()

    # Create a dummy input batch
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # ACT & ASSERT
    try:
        with torch.no_grad(): # for inference tests
            output = model(dummy_input)

        # Check that the output shape is correct: [Batch Size, Num Classes]
        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, f"Model output shape is incorrect for {model_name}"

    except Exception as e:
        pytest.fail(f"EfficientNet forward pass for {model_name} failed with an exception: {e}")
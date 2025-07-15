import torch
import torchvision
from torchvision import transforms
from PlantVision.data.transforms import get_transforms

def test_get_transforms_output_type():
    """Tests if the get_transforms function returns the correct object type"""

    # ARRANGE: Set up the inputs for the function
    test_img_size = 224

    # ACT: Call the function (get_transforms)
    result_transform = get_transforms(img_size=test_img_size)

    # ASSERT: Check if the result is what is expected
    assert isinstance(result_transform, torchvision.transforms.Compose), "The output should be a transforms.Compose object"

def test_get_transforms_resizes_correctly():
    """ Tests if the transform pipeline correctly resizes a dummy image tensor"""

    # ARRANGE
    test_img_size = 64

    # A dummy tensor representing a 3-channel image of a different size
    dummy_image = torch.randn(3, 100, 120) # (channels, height, width)

    # ACT
    transform_pipeline = get_transforms(img_size=test_img_size)
    transformed_image = transform_pipeline(dummy_image)

    # ASSERT: The output should be a tensor
    assert isinstance(transformed_image, torch.Tensor), "Transformed output should be a torch.Tensor"
    # Check the shape
    expected_shape = (3, test_img_size, test_img_size)
    assert transformed_image.shape == expected_shape, f"Image should be resized to {expected_shape}, but got {transformed_image.shape}"


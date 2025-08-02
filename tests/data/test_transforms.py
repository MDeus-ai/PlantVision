import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from PlantVision.data.transforms import get_transforms

def test_get_transforms_output_type():
    """Tests if the get_transforms function returns the correct object type"""

    # ARRANGE: Set up the inputs for the function
    test_img_size = 224
    test_mean = [0.485, 0.456, 0.406]
    test_std = [0.229, 0.224, 0.225]

    # ACT: Call the function (get_transforms)
    result_transform = get_transforms(img_size=test_img_size, mean=test_mean, std=test_std)

    # ASSERT: Check if the result is what is expected
    assert isinstance(result_transform, torchvision.transforms.Compose), "The output should be a transforms.Compose object"



def test_get_transforms_resizes_correctly():
    """ Tests if the transform pipeline correctly resizes a dummy image tensor"""

    # ARRANGE
    test_img_size = 64
    test_mean = [0.485, 0.456, 0.406]
    test_std = [0.229, 0.224, 0.225]

    # A dummy tensor representing a 3-channel image of a different size
    dummy_image_np = np.uint8(np.random.rand(100, 120, 3) * 255)
    dummy_pil_image = Image.fromarray(dummy_image_np)

    # ACT
    transform_pipeline = get_transforms(img_size=test_img_size, mean=test_mean, std=test_std)
    transformed_image = transform_pipeline(dummy_pil_image)

    # ASSERT: The output should be a tensor
    assert isinstance(transformed_image, torch.Tensor), "Transformed output should be a torch.Tensor"
    # Check the shape
    expected_shape = (3, test_img_size, test_img_size)
    assert transformed_image.shape == expected_shape, f"Image should be resized to {expected_shape}, but got {transformed_image.shape}"


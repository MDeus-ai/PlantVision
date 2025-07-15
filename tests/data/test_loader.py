import pytest
import torch
from pathlib import Path
from PIL import Image

from PlantVision.data.loader import get_dataloader
from PlantVision.data.transforms import get_transforms

@pytest.fixture(scope='session')
def dummy_dataset_path(tmpdir_factory):
    """
    A pytest fixture to create a temporary, fake dataset for testing.
    This fixture will only run once per test session
    """
    # Create a temporary base directory
    tmp_path = tmpdir_factory.mktemp('data')

    # Create two class directories (like 'apple', 'pineapple')
    class_a_path = Path(tmp_path) / 'class_a'
    class_b_path = Path(tmp_path) / 'class_b'
    class_a_path.mkdir()
    class_b_path.mkdir()

    # Create a few dummy image files in each class directory
    for i in range(5):
        # Create a small, black dummy image
        img = Image.new('RGB', (30, 30), color='black')
        img.save(class_a_path / f'a_{i}.png')
    for i in range(5):
        img = Image.new('RGB', (30, 30), color='white')
        img.save(class_b_path / f'a_{i}.png')

    # The fixture returns the path to the root of this fake dataset
    return Path(tmp_path)

def test_get_dataloader(dummy_dataset_path):
    """
    Tests if the get_dataloader function can create a dataloader and
    if the dataloader yields batches of the correct shape and type
    :param dummy_dataset_path:
    :return:
    """

    # ARRANGE
    test_img_size = 64
    test_batch_size = 4

    # Get the transform pipeline
    transforms = get_transforms(img_size=test_img_size)

    # ACT: Pass the path from the fixture to the dataloader function
    dataloader = get_dataloader(
        data_path=dummy_dataset_path,
        batch_size=test_batch_size,
        num_workers=0,
        transform=transforms,
    )

    # Get one batch from the dataloader
    images, labels = next(iter(dataloader))

    # ASSERT: Check that the batch of images is a tensor
    assert isinstance(images, torch.Tensor)
    # ASSERT: Check that the batch of labels is a tensor
    assert isinstance(labels, torch.Tensor)

    # Check the shape of the images batch: [Batch Size, Channels, Height, Width]
    expected_image_shape = (test_batch_size, 3, test_img_size, test_img_size)
    assert images.shape == expected_image_shape

    # Check the shape of the labels batch
    assert labels.shape == (test_batch_size, )

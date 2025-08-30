import torch
from plantvision.data.loader import get_dataloader
from plantvision.data.transforms import get_transforms


def test_get_dataloader(dummy_evaluation_project):
    """
    Tests the get_dataloader function using the shared dummy dataset.
    """

    # ARRANGE
    test_img_size = 64
    test_batch_size = 2
    test_mean = [0.485, 0.456, 0.406]
    test_std = [0.229, 0.224, 0.225]

    transforms = get_transforms(img_size=test_img_size, mean=test_mean, std=test_std)

    # Use the data_path from the dummy_evaluate_project fixture
    data_path = dummy_evaluation_project["data_path"]

    # ACT
    dataloader = get_dataloader(
        data_path=data_path,
        batch_size=test_batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        transform=transforms
    )
    images, labels = next(iter(dataloader))

    # ASSERT
    assert isinstance(images, torch.Tensor)
    assert labels.shape == (test_batch_size,)
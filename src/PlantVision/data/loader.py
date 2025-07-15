from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from pathlib import Path


def get_dataloader(data_path: Path, batch_size: int, num_workers: int, transform: callable):
    """
    Creates a DataLoader for image folder dataset.
    Args:
        data_path (Path): Path to the training data directory.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of CPU workers for loading data.
        transform (callable): The transformations to apply to the images.
    Returns:
        A torch.utils.data.DataLoader object.
    """
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
from torchvision import transforms

def get_transforms(img_size: int, mean: list, std: list):
    """
    Creates a transform pipeline.
    Args:
        img_size (int): The target image size.
        mean (list): A list of mean values.
        std (list): A list of std values.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
from torchvision import transforms

def get_transforms(img_size: int):
    """
    Creates a transform pipeline.
    Args:
        img_size (int): The target image size.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
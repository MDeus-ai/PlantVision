import pytest
import yaml
import torch
import json
import numpy as np
from PIL import Image
from PlantVision.models.efficientnet.EfficientNet import EfficientNet


@pytest.fixture(scope="session")
def dummy_evaluation_project(tmp_path_factory):
    """
    Creates a full temporary project structure for an end-to-end test and evaluation run
    ============================================================================
    eval_project_root/
    ├── data/
    │   └── val/
    │       ├── class_a/
    │       │    ├── 1.png
    │       │    └── 2.png
    │       ├── class_b
    │       │    ├── 3.png
    │       │    └── 4.png
    │       ├── class_c
    │       │    ├── 5.png
    │       │    └── 6.png
    │       ├── class_d
    │       │    ├── 7.png
    │       │    └── 8.png
    │       └── class_e
    │            ├── 9.png
    │            └── 10.png
    ├── outputs/
        ├── class_names.json
    │   └── best_model.pth
    └── configs/
        ├── data_config.yaml
        ├── model_config.yaml
        └── train_config.yaml
    ============================================================================
    """

    # Use tmp_path_factory to get a pathlib.Path object
    base_dir = tmp_path_factory.mktemp("eval_project_root")

    # 1. Create fake data in an appropriate structure
    data_dir = base_dir / "data" / "val"
    data_dir.mkdir(parents=True)
    (data_dir / "class_a").mkdir()
    (data_dir / "class_b").mkdir()
    (data_dir / "class_c").mkdir()
    (data_dir / "class_d").mkdir()
    (data_dir / "class_e").mkdir()


    # Create a dummy image for each class and save them in their respective class folders
    Image.fromarray(np.uint8(np.zeros((10, 10, 3)))).save(data_dir / "class_a" / "1.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_a" / "2.png")
    Image.fromarray(np.uint8(np.zeros((10, 10, 3)))).save(data_dir / "class_b" / "3.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_b" / "4.png")
    Image.fromarray(np.uint8(np.zeros((10, 10, 3)))).save(data_dir / "class_c" / "5.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_c" / "6.png")
    Image.fromarray(np.uint8(np.zeros((10, 10, 3)))).save(data_dir / "class_d" / "7.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_d" / "8.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_e" / "9.png")
    Image.fromarray(np.uint8(np.ones((10, 10, 3)))).save(data_dir / "class_e" / "10.png")


    # 2. Create a fake model checkpoint
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir()
    model = EfficientNet(
        num_classes=5,
        model_name='b0',
        pretrained=False,
        freeze_layers=False
    )
    model_path = outputs_dir / "best_model.pth"
    torch.save(model, model_path) # Save model state

    # Create a fake class_names.json
    class_names_path = outputs_dir / "class_names.json"
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=4)

    # 3. Create config files
    config_dir = base_dir / "configs"
    config_dir.mkdir()

    # Data Config
    data_config_path = config_dir / "data_config.yaml"
    data_config_data = {
        'data':
            {
             'train_dir': 'val',
             'batch_size': 2,
             'img_size': 64,
             'shuffle': False,
             'drop_last': True,
             'loader_num_workers': 0,
             'mean': [0.229, 0.224, 0.225],
             'std': [0.485, 0.456, 0.406]
             }
    }
    with open(data_config_path, 'w') as f:
        yaml.dump(data_config_data, f)

    # Model Config
    model_config_path = config_dir / "model_config.yaml"
    model_config_data = {
        'model':
            {'model_variation': 'b0',
             'num_classes': 5,
             'pretrained': False
             }
    }
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config_data, f)

    # Train Config
    train_config_path  = config_dir / "train_config.yaml"
    train_config_data = {
        'train':
            {'num_epochs': 5,
             'learning_rate': 0.001,
            }
    }
    with open(train_config_path, 'w') as f:
        yaml.dump(train_config_data, f)

    return {
        "project_root": base_dir,
        "model_path": model_path,
        "config_dir": config_dir,
        "data_path": data_dir,
        "data_config_path": data_config_path,
        "model_config_path": model_config_path,
        "train_config_path": train_config_path,
        "class_names_path": class_names_path,
    }
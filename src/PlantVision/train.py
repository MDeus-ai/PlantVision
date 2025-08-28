import json
import tqdm
import torch
import yaml
import shutil
import mlflow.pytorch
import torch.nn as nn
from pathlib import Path
import torchvision.datasets as datasets

from PlantVision import paths
from PlantVision.utils import load_config
from PlantVision.data.loader import get_dataloader
from PlantVision.data.transforms import get_transforms
from PlantVision.models.efficientnet.EfficientNet import EfficientNet

# To run the train.py script:
#   0. Open the project from the terminal
#   1. Activate PlantVision project environment by running: venv\Scripts\activate
#   2. Run the following command: python -m PlantVision.train


def main():
    """Main training script orchestration."""

    # Configuration Loading
    print("\n\n 🚀 Loading configurations...")
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")
    train_config = load_config(paths.CONFIG_DIR / "train_config.yaml")

    # Path to the training dataset
    train_data_path = paths.DATA_DIR / data_config['data']['train_dir']

    # Data Preparation
    print(" 📝 Preparing data...")
    train_transforms = get_transforms(
        img_size=data_config['data']['img_size'],
        mean=data_config['data']['mean'],
        std=data_config['data']['std']
    )

    # Temporary dataset instance to extract class names
    print(" ⛏️ Extracting class names from training data...")
    temp_dataset = datasets.ImageFolder(train_data_path)
    class_names = temp_dataset.classes

    # Write the list of class names to the JSON file
    class_names_path = paths.PROJECT_ROOT / "outputs" / "class_names.json"
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f" ⛏️ Saved {len(class_names)} class names to {class_names_path}")

    # Raise an exception if the num of classes in config doesn't...
    # match those in the class_names.json file
    model_num_classes = model_config['model']['num_classes']
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    assert len(class_names) == model_num_classes, \
        (f"class_names.json has {len(class_names)} classes but "
         f"num_classes in model_config.yaml has {model_num_classes} classes")


    train_loader = get_dataloader(
        data_path=train_data_path, # Path to the training dataset
        batch_size=data_config['data']['batch_size'],
        num_workers=data_config['data']['loader_num_workers'],
        shuffle=data_config['data'].get('shuffle', True),
        drop_last=data_config['data'].get('drop_last', True),
        transform=train_transforms
    )

    # Model Initialization
    model_params = model_config['model']


    # Selects a device onto which to perform the evaluation, GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" 💡 Using device: {device}\n")

    # Instantiate the model, passing all the necessary parameters from the config
    model = EfficientNet(
        num_classes=model_params['num_classes'],
        model_name=model_params['model_variation'],
        pretrained=model_params.get('pretrained', False),
        freeze_layers=model_params.get('freeze_layers', False)
    )
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['train']['learning_rate']
    )

    # MLflow and Training Loop
    mlflow.set_experiment('PlantVision Model Training')
    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")

        # Log all parameters from all configs
        mlflow.log_params(data_config['data'])
        mlflow.log_params(model_config['model'])
        mlflow.log_params(train_config['train'])
        mlflow.log_artifact(str(class_names_path), "class_names")

        print('\n ⚙️ Starting model training...')
        num_epochs = train_config['train']['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for imgs, labels in progress_bar:

                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Avg Training Loss: {epoch_loss:.4f}")
            mlflow.log_metric('train_loss', epoch_loss, step=epoch)

        print('\n ☑️ Training finished. Saving artifacts...')

        # 1. Save the best model state_dict locally for direct access
        output_dir = paths.PROJECT_ROOT / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = output_dir / 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f" ✅ Model state_dict saved locally to: {best_model_path}")

        # 2. Manually create an MLflow Model package
        print("Logging complete model package to MLflow...")

        # Define a path within the current run's artifact location
        # This will create a temporary local directory that gets uploaded automatically
        mlflow_model_path = "mlflow_model"

        # a. Save the state_dict inside the package's data subfolder
        model_data_path = Path(mlflow_model_path) / "data"
        model_data_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_data_path / "model.pth")

        # b. Get a model signature
        input_example, _ = next(iter(train_loader))
        signature = mlflow.models.infer_signature(input_example.cpu().numpy())

        # c. Manually create the MLmodel metadata file
        mlflow_model_metadata = {
            "artifact_path": "model",  # The name when loaded via mlflow.pytorch.load_model
            "flavors": {
                "pytorch": {
                    "model_data": "data",
                    "pytorch_version": torch.__version__,
                    "model_state_dict": "model.pth"
                }
            },
            "signature": signature.to_dict()
        }

        with open(Path(mlflow_model_path) / "MLmodel", "w") as f:
            yaml.dump(mlflow_model_metadata, f)

        # d. Log the requirements file with the model
        shutil.copyfile("requirements.txt", Path(mlflow_model_path) / "requirements.txt")

        # 3. Log the entire manually created directory as an artifact
        mlflow.log_artifacts(local_dir=mlflow_model_path, artifact_path="model")

        print(f'✅ Successfully logged model package to MLflow Run ID: {run.info.run_id}')

        # Clean up the temporary local directory
        shutil.rmtree(mlflow_model_path)

if __name__ == '__main__':
    main()
import yaml
import tqdm
import torch
import mlflow.pytorch
import torch.nn as nn
from src import paths
from PlantVision.data.transforms import get_transforms
from PlantVision.data.loader import get_dataloader
from PlantVision.models.efficientnet.EfficientNet import EfficientNet

# To run the train.py script:
#   0. Open the project from the terminal
#   1. Activate PlantVision project environment by running: venv\Scripts\activate
#   2. Run the following command: python -m PlantVision.train

def load_config(config_path):
    """A single, reusable function to load any YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training script orchestration."""

    # Configuration Loading
    print("\n\n 🚀 Loading configurations...")
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")
    train_config = load_config(paths.CONFIG_DIR / "train_config.yaml")

    # Data Preparation
    print(" 📝 Preparing data...")
    train_transforms = get_transforms(
        img_size=data_config['data']['img_size'],
        mean=data_config['data']['mean'],
        std=data_config['data']['std']
    )

    train_loader = get_dataloader(
        data_path=paths.DATA_DIR / data_config['data']['train_dir'], # Path to the training dataset
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

        # A sample for the model signature
        input_example, _ = next(iter(train_loader))
        numpy_input_example = input_example.cpu().numpy()

        print('\n ☑️ Training finished. Logging model to MLflow...')
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path='model',
            input_example=numpy_input_example,
        )
        print(f'\n ✅ Successfully logged model to MLflow Run ID: {run.info.run_id}')


if __name__ == '__main__':
    main()
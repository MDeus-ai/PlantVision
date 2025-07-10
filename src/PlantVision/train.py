import yaml
import tqdm
import torch
import mlflow.pytorch
import torch.nn as nn

from src import paths
from PlantVision.data.transforms import get_transforms
from PlantVision.data.loader import get_dataloader
from PlantVision.models.efficientnet.EfficientNet import EfficientNet


def load_config(config_path):
    """A single, reusable function to load any YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training script orchestration."""
    # --- 1. Configuration Loading (The ONLY place we read files) ---
    print("Loading configurations...")
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")
    train_config = load_config(paths.CONFIG_DIR / "train_config.yaml")

    # --- 2. Data Preparation (Dependency Injection in action) ---
    print("Preparing data...")
    train_transforms = get_transforms(img_size=data_config['data']['img_size'])

    train_loader = get_dataloader(
        data_path=paths.DATA_DIR / data_config['data']['train_dir'],
        batch_size=data_config['data']['batch_size'],
        num_workers=data_config['data']['loader_num_workers'],
        transform=train_transforms
    )

    # --- 3. Model Initialization ---
    print(f"Initializing model: EfficientNet-{model_config['model']['model_variation']}")
    model = EfficientNet(model_name=model_config['model']['model_variation'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['train']['learning_rate']
    )

    # --- 4. MLflow and Training Loop ---
    mlflow.set_experiment('PlantVision Model Training')
    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")

        # Log all parameters from all configs
        mlflow.log_params(data_config['data'])
        mlflow.log_params(model_config['model'])
        mlflow.log_params(train_config['train'])

        print('Starting model training...')
        num_epochs = train_config['train']['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for imgs, labels in progress_bar:
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

        print('Training finished. Logging model to MLflow...')
        mlflow.pytorch.log_model(model, 'model')
        print(f'\n ☑️ Successfully logged model to MLflow Run ID: {run.info.run_id}')


if __name__ == '__main__':
    main()
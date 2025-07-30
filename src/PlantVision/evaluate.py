import torch
import yaml
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from src import paths
from PlantVision.data.loader import get_dataloader
from PlantVision.data.transforms import get_transforms

# To run the evaluate.py script
# COMMANDS

# # Run with default paths
# python -m PlantVision.evaluate
#
# # Or specify a different model and dataset
# python -m PlantVision.evaluate --model-checkpoint "mlruns/0/.../artifacts/model/model.pth" --data-path "data/processed/test"

# A helper function to open and read config files
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

#=======================================================================
def evaluate(model_checkpoint: Path, data_path: Path, data_config_path: Path):
    """
    Evaluates a trained model checkpoint on a given dataset

    Steps:
    1. Loads the model and the data.
    2. Runs every image in the dataset through the model to get predictions.
    3. Compares the model's predictions to the true labels.
    4. Calculates performance metrics (precision, recall, F1-score).
    5. Saves a text-based classification_report.txt and a visual confusion_matrix.png to the project's outputs directory.

    Args:
        model_checkpoint (Path): Path to the saved .pth model checkpoint
        data_path (Path): Path to the validation/test data directory
        data_config_path (Path): Path to the main data configuration file
    """
    print("Starting Evaluation...\n")

    # 1. Load Configurations
    print(f"Loading data config from: {data_config_path}")
    # Reads from the data_config.yaml file
    data_config = load_config(data_config_path)['data']

    # Selects a device onto which to perform the evaluation, GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 2. Prepare Dataset and DataLoader
    print("Preparing validation dataloader...\n")
    img_size = data_config['img_size']
    batch_size = data_config['batch_size']
    mean = data_config['mean']
    std = data_config['std']


    # Use the same transforms as validation/training
    val_transforms = get_transforms(img_size=img_size, mean=mean, std=std)

    # Create a temporary dataset to get class names, then the dataloader
    temp_dataset = datasets.ImageFolder(data_path)
    class_names = temp_dataset.classes

    # Validation dataloader
    val_loader = get_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=0,
        transform=val_transforms,
        shuffle=False,
        drop_last=False,
    )

    # 3. Load the model & state dictionary
    print(f"Loading model checkpoint from: {model_checkpoint}")
    model = torch.load(model_checkpoint, map_location=device, weights_only=False)
    model.to(device)
    model.eval() # Set to evaluation mode

    # 4. Run Inference and Collect Predictions
    print("Running inference on the dataset...\n")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate and Display Metrics
    print("\n ======> Evaluation Results <======")

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print("Classification Report:")
    print(report)

    # Save the report to a file
    report_path = paths.PROJECT_ROOT / "outputs" / "classification_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Greens')
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Save the confusion matrix plot
    cm_path = paths.PROJECT_ROOT / "outputs" / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    print("\n=====> Evaluation Finished <=====")


if __name__ == "__main__":
    # CLI functionality of the evaluate.py script
    parser = argparse.ArgumentParser(description='Evaluate a trained PlantVision model')

    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="outputs/best_model.pth", # Default path relative to project root
        help="Path to the trained model .pth file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/val", # Default path to a validation set
        help="Path to the validation/test dataset directory."
    )
    parser.add_argument(
        "--data-config-path",
        type=str,
        default="configs/data_config.yaml",
        help="Path to the data configuration YAML file."
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yaml",
        help="Path to the model configuration YAML file."
    )

    args = parser.parse_args()

    # Construct absolute paths from the project root
    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    data_path = paths.PROJECT_ROOT / args.data_path
    data_config_path = paths.PROJECT_ROOT / args.data_config_path

    evaluate(
        model_checkpoint=model_path,
        data_path=data_path,
        data_config_path=data_config_path,
    )

import torch
import yaml
import tqdm
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

# To run the evaluate.py script:
#   0. Open the project from the terminal
#   1. Activate PlantVision project environment by running: venv\Scripts\activate
#   2. Run a sample evaluation command: python -m PlantVision.evaluate --model-checkpoint "outputs/best_model.pth" --data-path "data/processed/val"

# Flags:
#   --model-checkpoint : Path to the trained model .pth file (defaults to /outputs/best_model.pth)
#   --data-path : Path to the validation or test dataset directory
#   --data-config-path : Path to the data configuration YAML file (defaults to /configs/data_config.yaml)
#   --model-config-path : Path to the data configuration YAML file (defaults to /configs/model_config.yaml)

# For more check out the documentation https://github.com/MDeus-ai/PlantVision


# A helper function to open and read config files
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(model_checkpoint: Path, data_path: Path, data_config_path: Path):
    """
    Evaluates a trained model checkpoint on a given dataset

    Steps:
    1. Loads configurations, model architecture, and dataset.
    2. Loads the trained weights into the model.
    3. Runs inference on the dataset to get predictions.
    4. Calculates Top-1, Top-2 and Top-5 accuracy.
    5. Calculates and saves a detailed classification report and confusion matrix.

    Args:
        model_checkpoint (Path): Path to the saved .pth model checkpoint
        data_path (Path): Path to the validation/test data directory
        data_config_path (Path): Path to the main data configuration file
    """
    print("\n"*3 + " Starting Evaluation...\n")

    # 1. Load Configurations
    print(f"🚀 Loading data config from: {data_config_path}")
    # Reads from the data_config.yaml file
    data_config = load_config(data_config_path)['data']

    # Selects a device onto which to perform the evaluation, GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💡 Using device: {device}")

    # 2. Prepare Dataset and DataLoader
    # print("Preparing validation dataloader...\n")
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
    print(f"🔁 Loading model checkpoint from: {model_checkpoint}")
    model = torch.load(model_checkpoint, map_location=device, weights_only=False)
    model.to(device)
    model.eval() # Set to evaluation mode

    # 4. Run Inference and Collect Predictions
    all_preds_top1 = []
    all_labels = []

    correct_top1 = 0
    correct_top2 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(val_loader, desc="🧠 Evaluating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Top-1 accuracy calculation
            _, preds_top1 = torch.max(outputs, 1)
            all_preds_top1.extend(preds_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct_top1 += (preds_top1 == labels).sum().item()
            total += labels.size(0)

            # Top-2 accuracy calculation
            _, preds_top2 = torch.topk(outputs, 2, dim=1)
            labels_reshaped_t2 = labels.view(-1, 1)
            correct_top2 += (preds_top2 == labels_reshaped_t2).sum().item()

            # Top-5 accuracy calculation
            _, preds_top5 = torch.topk(outputs, 5, dim=1)
            labels_reshaped_t5 = labels.view(-1, 1)
            correct_top5 += (preds_top5 == labels_reshaped_t5).sum().item()

    # 5. Calculate and Display Metrics
    # Draw a box around the PlantVision Prediction word
    top1_accuracy = (correct_top1 / total) * 100
    top2_accuracy = (correct_top2 / total) * 100
    top5_accuracy = (correct_top5 / total) * 100

    title = "Evaluation Results"
    box_width = len(title) + 4
    print("\n"*3 + "\t"*5 + "┌" + "─" * box_width + "┐")
    print("\t"*5 + "│  " + title + "  │")
    print("\t"*5 + "└" + "─" * box_width + "┘")

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds_top1, target_names=class_names, digits=4, zero_division=0)
    print("\n\n" + "\t"*5 + " Classification Report (Top-1):\n")
    print(report)

    # Display Top-1, Top-2 and Top-5 Accuracies
    print("\t"*5 + "="*50)
    print("\t"*5 + f"🎯 Top-1 Accuracy: {top1_accuracy:.2f}%")
    print("\t"*5 + f"📈 Top-2 Accuracy: {top2_accuracy:.2f}%")
    print("\t"*5 + f"📉 Top-5 Accuracy: {top5_accuracy:.2f}%")

    # Save the report to a file
    report_path = paths.PROJECT_ROOT / "outputs" / "classification_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"Top-1 Accuracy: {top1_accuracy:.2f}%\n")
        f.write(f"Top-2 Accuracy: {top2_accuracy:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.2f}%\n")
    print("\n"*3 + f" Classification report saved to {report_path}")

    # Confusion Matrix
    print(" Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds_top1)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Greens')
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save the confusion matrix plot
    cm_path = paths.PROJECT_ROOT / "outputs" / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f" Confusion matrix plot saved to {cm_path}")
    print("\n"*3 + "Evaluation Finished." + "\n"*3)


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
        default=None,
        help="Path to the validation or test dataset directory."
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

    config_path = paths.PROJECT_ROOT / args.data_config_path

    if args.data_path is None:
        data_config = load_config(config_path)
        final_data_path = paths.DATA_DIR / data_config['data']['val_dir']
    else:
        final_data_path = paths.PROJECT_ROOT / args.data_path

    # Construct absolute paths from the project root
    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    data_path = final_data_path
    data_config_path = paths.PROJECT_ROOT / args.data_config_path

    evaluate(
        model_checkpoint=model_path,
        data_path=data_path,
        data_config_path=data_config_path,
    )

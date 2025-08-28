import tqdm
import json
import torch
import mlflow
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from PlantVision import paths
from PlantVision.utils import load_config
from PlantVision.data.loader import get_dataloader
from PlantVision.data.transforms import get_transforms
from PlantVision.models.efficientnet.EfficientNet import EfficientNet

# To run the evaluate.py script:
#   0. Open the PlantVision project from the terminal
#   1. Activate PlantVision project environment by running: venv\Scripts\activate
#   2. Run a sample evaluation command: python -m PlantVision.evaluate --model-checkpoint "outputs/best_model.pth" --data-path "data/processed/val"

# Flags:
#   --model-checkpoint : Path to the trained model .pth file (defaults to /outputs/best_model.pth)
#   --data-path : Path to the validation or test dataset directory in the project

# For more check out the documentation https://github.com/MDeus-ai/PlantVision


def evaluate(model_checkpoint: Path, data_path: Path, run_id: str=None):
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
        run_id (str, optional): The MLflow Run ID to log metrics to.
    """
    print("\n"*3 + " Starting Evaluation...\n")

    # 1. Load Configurations
    # Reads from the data_config.yaml file
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")['data']

    # Selects a device onto which to perform the evaluation, GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💡 Using device: {device}")

    # 2. Prepare Dataset and DataLoader
    # print("Preparing validation dataloader...\n")
    img_size = data_config['img_size']
    batch_size = data_config['batch_size']
    mean = data_config.get('mean')
    std = data_config.get('std')


    # Use the same transforms as validation/training
    val_transforms = get_transforms(img_size=img_size, mean=mean, std=std)

    # Load class names from the class_names.json file
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")
    class_names_path = paths.PROJECT_ROOT / "outputs" / "class_names.json"
    print(f"📑 Loading class names from: {class_names_path}")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"📑 Found {num_classes} classes")

    # Raise an exception if the num of classes in config doesn't...
    # match those in the class_names.json file
    model_num_classes = model_config['model']['num_classes']
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    assert len(class_names) == model_num_classes, \
        (f"class_names.json has {len(class_names)} classes but "
         f"num_classes in model_config.yaml has {model_num_classes} classes")

    # Validation dataloader
    val_loader = get_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=0,
        transform=val_transforms,
        shuffle=False,
        drop_last=False,
    )

    # 3. Load the model, configs & state dictionary (weights)
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")["model"]
    print(f"🔁 Loading model checkpoint from: {model_checkpoint}")
    model = EfficientNet(
        num_classes=model_num_classes,
        model_name=model_config["model_variation"],
        pretrained=False,
    )
    # Load model weights
    state_dict = torch.load(model_checkpoint, map_location=device)
    # Load the weights into the model instance
    model.load_state_dict(state_dict)
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
    print("\n"*3 + "\t"*7 + "┌" + "─" * box_width + "┐")
    print("\t"*7 + "│  " + title + "  │")
    print("\t"*7 + "└" + "─" * box_width + "┘")

    # Classification Report (Precision, Recall, F1-Score)
    report_dict = classification_report(
        all_labels,
        all_preds_top1,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    report_text = classification_report(
        all_labels,
        all_preds_top1,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\n\n" + "\t"*7 + " Classification Report (Top-1):\n")
    print(report_text)

    # Display Top-1, Top-2 and Top-5 Accuracies
    print("\t"*5 + "="*51)
    print("\t"*5 + f"🎯 Top-1 Accuracy: {top1_accuracy:.2f}%")
    print("\t"*5 + f"📈 Top-2 Accuracy: {top2_accuracy:.2f}%")
    print("\t"*5 + f"📉 Top-5 Accuracy: {top5_accuracy:.2f}%")

    # Save the report to a file
    report_path = paths.PROJECT_ROOT / "outputs" / "classification_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
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

    if run_id:
        print(f"\n 🔄️ Logging evaluation results to MLflow Run ID: {run_id}")
        with mlflow.start_run(run_id=run_id):
            # Log summary metrics
            mlflow.log_metric("eval_accuracy", report_dict["accuracy"])
            mlflow.log_metric("eval_macro_avg_precision", report_dict["macro avg"]["precision"])
            mlflow.log_metric("eval_macro_avg_recall", report_dict["macro avg"]["recall"])
            mlflow.log_metric("eval_macro_avg_f1-scroe", report_dict["macro avg"]["f1-score"])

            # Log the output files as artifacts
            mlflow.log_artifact(str(report_path), "evaluation_reports")
            mlflow.log_artifact(str(cm_path), "evaluation_report")

            mlflow.set_tag("status", "evaluated")



def main():
    # CLI functionality of the evaluate.py script
    parser = argparse.ArgumentParser(description='Evaluate a trained PlantVision model')

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="The MLflow Run ID of the training run to log evaluation results to."
    )
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

    args = parser.parse_args()

    if args.data_path is None:
        data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")
        final_data_path = paths.DATA_DIR / data_config['data']['val_dir']
    else:
        final_data_path = paths.PROJECT_ROOT / args.data_path

    # Construct absolute paths from the project root
    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    data_path = final_data_path

    evaluate(
        model_checkpoint=model_path,
        data_path=data_path,
        run_id=args.run_id,
    )

if __name__ == "__main__":
    main()
import torch
import yaml
import argparse
from pathlib import Path
from PIL import Image
import torchvision.datasets as datasets

from src import paths
from PlantVision.data.transforms import get_transforms

def predict(model_checkpoint: Path, image_path: Path, data_config_path: Path, class_data_dir: Path):
    """
    Runs inference on a single image using a trained model checkpoint

    Args:
        model_checkpoint (Path): Path to the saved .pth model checkpoint
        image_path (Path): Path to the single image file for prediction
        data_config_path (Path): Path to the data configuration file to get img_size
        class_data_dir (Path): Path to the training data directory to infer class names
    """

    if not image_path.is_file():
        print(f"Error: Image file not found at {image_path}")
        return

    print("\n =====> PlantVision Prediction <=====")
    print(f" 🖼️ Image: {image_path}")

    # 1. Load Configurations and Class Names
    # Load data configurations
    data_config = yaml.safe_load(data_config_path.read_text())["data"]
    img_size = data_config["img_size"]
    mean = data_config["mean"]
    std = data_config["std"]

    # Infer class names from the training directory structure
    temp_dataset = datasets.ImageFolder(class_data_dir)
    class_names = temp_dataset.classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" 💡 Using device: {device}")

    # 2. Load Model
    print(f" 🔁 Loading model checkpoint...")
    model = torch.load(model_checkpoint, map_location=device, weights_only=False)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # 3. Load and Preprocess the Image: Using the same transformations used during validation
    transforms = get_transforms(img_size=img_size, mean=mean, std=std)

    # Load the image using Pillow
    img = Image.open(image_path).convert("RGB")

    # Apply transforms and add a batch dimension (B, C, H, W)
    input_tensor = transforms(img).unsqueeze(0).to(device)

    # 4. Run Inference
    print("\n 🧠 Running inference...")
    with torch.no_grad():
        output = model(input_tensor)

    # 5. Process Output
    # Apply softmax to get probabilities from logits
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top prediction
    top_prob, top_catid = torch.max(probabilities, dim=0)
    predicted_class = class_names[top_catid]
    confidence = top_prob.item()

    # 6. Display Results
    print(f"\n 🌿 Prediction: {predicted_class}")
    print(f" 🔢 Confidence: {confidence:.4f} ({confidence:.2%})")
    print("\n Model Exited.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image with a trained PlantVision model.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="outputs/best_model.pth",
        help="Path to the trained model .pth file, relative to project root."
    )
    parser.add_argument(
        "--class-data-dir",
        type=str,
        default="data/processed/train",  # The directory used to get class names
        help="Path to the training data directory to infer class names, relative to project root."
    )
    parser.add_argument(
        "--data-config-path",
        type=str,
        default="configs/data_config.yaml",
        help="Path to the data configuration YAML file, relative to project root."
    )

    args = parser.parse_args()

    # Construct absolute paths using our reliable paths.py module
    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    image_path = Path(args.image).resolve()  # Use resolve for user-provided paths
    class_data_path = paths.PROJECT_ROOT / args.class_data_dir
    data_config_path = paths.PROJECT_ROOT / args.data_config_path

    predict(
        model_checkpoint=model_path,
        image_path=image_path,
        data_config_path=data_config_path,
        class_data_dir=class_data_path
    )

    # COMMANDS
    # # Example using a relative path to an image in an "assets" folder
    # python -m PlantVision.predict --image "D:\OTHERS\ME\me.jpg"

    # # Example using an absolute path to an image on your Desktop
    # python -m PlantVision.predict --image "C:/Users/YourUser/Desktop/test_plant.png"

    # # Example specifying a different model checkpoint
    # python -m PlantVision.predict --image "assets/sample_leaf.jpg" --model-checkpoint "mlruns/0/some_run_id/artifacts/model/model.pth"
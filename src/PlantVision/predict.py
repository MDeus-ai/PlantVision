import json
import torch
import yaml
import argparse
from pathlib import Path
from PIL import Image

from src import paths
from PlantVision.data.transforms import get_transforms

# To use predict to make inference on an image anywhere on the computer:
#   0. Open the PlantVision project from the terminal
#   1. Activate PlantVision project environment by running: venv\Scripts\activate
#   2. Command to make a prediction: python -m PlantVision.predict --image "path/to/image.png" --model-checkpoint "/outputs/best_model.pth"

# Flags:
#   --image: Path to any image on the computer
#   --model-checkpoint : Path to the trained model .pth file (defaults to /outputs/best_model.pth)
#   --verbose or -v : Print out more details about the model's predictions.

# For more check out the documentation https://github.com/MDeus-ai/PlantVision



def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict(model_checkpoint: Path, image_path: Path, verbose: bool = False):
    """
    Runs inference on a single image using a trained model checkpoint
    Optionally displays the top-10 most likely predictions for each image in verbose mode

    Args:
        model_checkpoint (Path): Path to the saved .pth model checkpoint
        image_path (Path): Path to the single image file for prediction
        verbose (bool, optional): If True, prints out dense inference results
    """

    if not image_path.is_file():
        print("\n" + "\t"*5 + f"❌ Error: Image file not found at {image_path}")
        return

    # Draw a box around the PlantVision Prediction word
    heading = "PlantVision Prediction"
    box_width = len(heading) + 4
    print("\n"*3 + "\t"*5 + "┌" + "─" * box_width + "┐")
    print("\t"*5 + "│  " + heading + "  │")
    print("\t"*5 + "└" + "─" * box_width + "┘")

    print("\n" + "\t"*5 + f"🖼️ Image: {image_path}")

    # 1. Load Configurations
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")["data"]
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")["model"]

    # Load data configurations
    img_size = data_config["img_size"]
    mean = data_config["mean"]
    std = data_config["std"]

    # Load class names from the class_names.json file
    class_names_path = paths.PROJECT_ROOT / "outputs" / "class_names.json"
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Raise an exception if the num of classes in config doesn't...
    # match those in the class_names.json file
    model_num_classes = model_config["num_classes"]
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    assert num_classes == model_num_classes, \
        (f"class_names.json has {len(class_names)} classes but "
         f"num_classes in model_config.yaml has {model_num_classes} classes")


    # Where to carry out inference from, GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "\t"*5 + f"💡 Using device: {device}")

    # 2. Load Model
    print("\n" + "\t"*5 + f"🔁 Loading model checkpoint...")
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
    print("\n" + "\t"*5 + "🧠 Running inference...")
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
    print("\n" + "\t"*5 + "="*30)
    print("\n" + "\t"*5 + f"🌿 Prediction: {predicted_class}")
    if verbose:
        print("\n" + "\t"*5 + f"🔢 Confidence: {confidence:.4f} ({confidence:.4%})")
    else:
        print("\n" + "\t"*5 + f"🔢 Confidence: {confidence:.0%}")

    # Display other likely predictions in verbose mode
    if verbose:
        print("\n\n" + "\t"*3 + "🏆 Top 10 Most Likely Predictions")
        print("\t"*3 + "-"*66)
        top10_prob, top10_catid = torch.topk(probabilities, 10)
        for i in range(top10_prob.size(0)):
            class_name = class_names[top10_catid[i]]
            prob = top10_prob[i].item()
            print("\n" + "\t"*3 + f"{i+1}. {class_name}" + "."*10 + f"Confidence: {prob:.4f} ({prob:.4%})")

        print("\n" * 3 + "\t" * 3 + "Model Exited." + "\n" * 3)
    else:
        print("\n"*3 + "\t"*5 + "Model Exited." + "\n"*3)

if __name__ == "__main__":
    # CLI functionality of the predict.py script
    parser = argparse.ArgumentParser(description="Run inference on a single image with a trained PlantVision model.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file from anywhere on the computer."
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="outputs/best_model.pth",
        help="Path to the trained model .pth file, relative to project root."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print out more details about the model's predictions."
    )
    args = parser.parse_args()

    # Construct absolute paths using our reliable paths.py module
    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    image_path = Path(args.image).resolve()  # Use resolve for user-provided paths
    data_config_path = paths.PROJECT_ROOT / args.data_config_path

    predict(
        model_checkpoint=model_path,
        image_path=image_path,
        verbose=args.verbose,
    )

import time
import tqdm
import json
import torch
import argparse
import numpy as np
import onnxruntime
from pathlib import Path
from sklearn.metrics import accuracy_score

from plantvision import paths
from plantvision.utils import load_config
from plantvision.data.loader import get_dataloader
from plantvision.data.transforms import get_transforms
from plantvision.models.efficientnet.EfficientNet import EfficientNet

def run_pytorch_evaluation(model, dataloader, device, desc=""):
    """Runs inference loop for a PyTorch model."""
    model.eval()
    all_preds, all_labels = [], []
    # Create the progress bar here
    progress_bar = tqdm.tqdm(dataloader, desc=desc)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def run_onnx_evaluation(session, dataloader, desc=""):
    """Runs inference loop for an ONNX model."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    all_preds, all_labels = [], []
    # Create the progress bar here
    progress_bar = tqdm.tqdm(dataloader, desc=desc)
    for images, labels in progress_bar:
        result = session.run([output_name], {input_name: images.numpy()})
        preds = np.argmax(result[0], axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

def benchmark_pytorch_model(model, img_size, device):
    """Benchmark inference latency for a PyTorch model."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    timings = []
    for _ in range(10): _ = model(dummy_input)
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            timings.append(end_time - start_time)
    return np.mean(timings) * 1000

def benchmark_onnx_session(session, img_size):
    """Benchmarks inference latency for an ONNX session."""
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    timings = []
    for _ in range(10): _ = session.run(None, {input_name: dummy_input}) # Warm-up
    for _ in range(100):
        start_time = time.time()
        _ = session.run(None, {input_name: dummy_input})
        end_time = time.time()
        timings.append(end_time - start_time)
    return np.mean(timings) * 1000


def validate_and_compare_all(pytorch_checkpoint: Path, onnx_base_path: Path, **kwargs):
    """
    Performs a head-to-head comparison of PyTorch, FP32 ONNX, and INT8 ONNX models.
    """
    print("\n"*3 + "Starting Full Model Bake-Off...\n")

    # Unpack keyword arguments
    data_path = kwargs['data_path']
    data_config_path = kwargs['data_config_path']
    model_config_path = kwargs['model_config_path']
    class_names_path = kwargs['class_names_path']

    # 1. Load Common Assets
    data_config = load_config(data_config_path)['data']
    model_config = load_config(model_config_path)['model']
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # 2. Prepare DataLoader
    img_size = data_config['img_size']
    val_transforms = get_transforms(img_size=img_size, mean=data_config['mean'], std=data_config['std'])
    val_loader = get_dataloader(
        data_path=data_path, batch_size=data_config['batch_size'],
        num_workers=0,
        transform=val_transforms,
        shuffle=False,
        drop_last=True
    )

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💡 Using device: {device}")

    # 3. Analyze Original PyTorch Model
    print("\n====> Analyzing PyTorch Model")
    pytorch_model = EfficientNet(num_classes=len(class_names), model_name=model_config['model_variation'])
    pytorch_model.load_state_dict(torch.load(pytorch_checkpoint, map_location=device))
    pytorch_model.to(device)
    results['pytorch'] = {
        'accuracy': run_pytorch_evaluation(pytorch_model, val_loader, device, desc="Validating PyTorch"),
        'size_mb': pytorch_checkpoint.stat().st_size / 1e6,
        'latency_ms': benchmark_pytorch_model(pytorch_model, img_size, device)
    }

    # 4. Analyze FP32 ONNX Model
    print("\n====> Analyzing FP32 ONNX Model")
    fp32_onnx_path = onnx_base_path.with_suffix(".fp32.onnx")
    fp32_session = onnxruntime.InferenceSession(str(fp32_onnx_path))
    results['fp32_onnx'] = {
        'accuracy': run_onnx_evaluation(fp32_session, val_loader, desc="Validating FP32 ONNX"),
        'size_mb': fp32_onnx_path.stat().st_size / 1e6,
        'latency_ms': benchmark_onnx_session(fp32_session, img_size)
    }

    # 5. Analyze INT8 ONNX Model
    print("\n====> Analyzing INT8 ONNX Model")
    int8_onnx_path = onnx_base_path.with_suffix(".int8.onnx")
    int8_session = onnxruntime.InferenceSession(str(int8_onnx_path))
    results['int8_onnx'] = {
        'accuracy': run_onnx_evaluation(int8_session, val_loader, desc="Validating INT8 ONNX"),
        'size_mb': int8_onnx_path.stat().st_size / 1e6,
        'latency_ms': benchmark_onnx_session(int8_session, img_size)
    }

    # 6. Print Comparison Summary
    print("\n"*3 + "=" * 80)
    print("                ✨ MODEL PERFORMANCE BAKE-OFF ✨")
    print("=" * 80)
    print(f"{'Metric':<25} | {'PyTorch (Original)':<20} | {'ONNX FP32':<15} | {'ONNX INT8':<15}")
    print("-" * 80)
    print(
        f"{'Accuracy':<25} | {results['pytorch']['accuracy']:<19.4f} | {results['fp32_onnx']['accuracy']:<14.4f} | {results['int8_onnx']['accuracy']:<14.4f}")
    print(
        f"{'Model Size (MB)':<25} | {results['pytorch']['size_mb']:<19.2f} | {results['fp32_onnx']['size_mb']:<14.2f} | {results['int8_onnx']['size_mb']:<14.2f}")
    print(
        f"{'Avg. Latency (ms)':<25} | {results['pytorch']['latency_ms']:<19.2f} | {results['fp32_onnx']['latency_ms']:<14.2f} | {results['int8_onnx']['latency_ms']:<14.2f}")
    print("=" * 80)

    # Calculate and print the final trade-off summary vs the original PyTorch model
    accuracy_drop = results['pytorch']['accuracy'] - results['int8_onnx']['accuracy']
    size_reduction = 1 - (results['int8_onnx']['size_mb'] / results['pytorch']['size_mb'])
    speed_improvement = results['pytorch']['latency_ms'] / results['int8_onnx']['latency_ms']

    print("\n🗒️ Summary of Trade-offs (INT8 vs Original PyTorch)")
    print(f"📉 Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop:.2%})")
    print(f"📉 Size Reduction: {size_reduction:.2%}")
    print(f"🚀 Inference Speed-up: {speed_improvement:.2f}x")
    print("\nValidation Complete...")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch, FP32 ONNX, and INT8 ONNX models.")

    parser.add_argument("--pytorch-checkpoint", type=str, required=True,
                        help="Path to the original trained PyTorch .pth file.")
    parser.add_argument("--onnx-base-path", type=str, required=True,
                        help="Base path to the exported ONNX models (e.g., 'models/plantvision_b0').")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--data-config-path", type=str, default="configs/data_config.yaml")
    parser.add_argument("--model-config-path", type=str, default="configs/model_config.yaml")
    parser.add_argument("--class-names-path", type=str, default="outputs/class_names.json")

    args = parser.parse_args()

    paths_dict = {
        "pytorch_checkpoint": paths.PROJECT_ROOT / args.pytorch_checkpoint,
        "onnx_base_path": paths.PROJECT_ROOT / args.onnx_base_path,
        "data_path": paths.PROJECT_ROOT / args.data_path,
        "data_config_path": paths.PROJECT_ROOT / args.data_config_path,
        "model_config_path": paths.PROJECT_ROOT / args.model_config_path,
        "class_names_path": paths.PROJECT_ROOT / args.class_names_path,
    }

    validate_and_compare_all(**paths_dict)


if __name__ == "__main__":
    main()
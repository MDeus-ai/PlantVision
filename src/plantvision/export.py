import onnx
import json
import torch
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_static, QuantType, quant_pre_process

from plantvision import paths
from plantvision.data.loader import get_dataloader
from plantvision.data.transforms import get_transforms
from plantvision.utils import load_config, PlantVisionDataReader
from plantvision.models.efficientnet.EfficientNet import EfficientNet


def export_model(model_checkpoint: Path, output_dir: Path, quantize: bool=False):
    """
    Exports a trained Pytorch model to ONNX format and optionally quantizes it

    Args:
    :param model_checkpoint: Path to the saved .pth model file:
    :param output_dir: Path to save the final .onnx model:
    :param quantize: If True, applies dynamic quantization to the model
    """

    print("⚙️ Initializing Model Export...")

    # 1. Load Configurations and Class Names
    print("🚀 Loading configs...")
    model_config = load_config(paths.CONFIG_DIR / "model_config.yaml")["model"]
    data_config = load_config(paths.CONFIG_DIR / "data_config.yaml")["data"]

    with open(paths.OUTPUTS_DIR / "class_names.json", 'r') as f:
        class_names = json.load(f)


    model_variation = model_config["model_variation"]

    img_size = 224

    # 2. Instantiate Model and Load Weights
    print(f"🔄️ Instantiating EfficientNet-{model_variation} for export...")
    model = EfficientNet(
        num_classes=len(class_names),
        model_name=model_variation,
        pretrained=False,
        freeze_layers=False
    )
    model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))
    model.eval()

    # 3. Export to ONNX (FP32)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"plantvision_{model_variation}"
    onnx_fp32_path = output_dir / f"{base_filename}.fp32.onnx"

    # Create a dummy tensor, required by ONNX exporter to trace the model's computation graph
    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=False)
    print(f"\n📦 Exporting model to ONNX (FP32) at: {onnx_fp32_path}")

    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=onnx_fp32_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print("☑️ FP32 ONNX export complete!")

    # 5. Apply Static Quantization
    if quantize:

        # Create a calibration dataloader
        print(f"\n🔧 Initializing Static Quantization...")

        # Pre-process the FP32 model
        preprocessed_fp32_path = output_dir / f"{base_filename}.preprocessed.onnx"
        print(f"🔄️ Pre-processing mode and saving to: {preprocessed_fp32_path}")
        quant_pre_process(
            input_model_path=onnx_fp32_path,
            output_model_path=preprocessed_fp32_path
        )

        calib_transforms = get_transforms(img_size=img_size, mean=data_config['mean'], std=data_config['std'])
        calib_data_path = paths.DATA_DIR / data_config.get("val_dir", data_config["train_dir"])
        calib_dataloader = get_dataloader(
            data_path=calib_data_path,
            batch_size=8,
            num_workers=0,
            transform=calib_transforms,
            shuffle=True,
            drop_last=False,
        )

        # Create a data reader
        onnx_model = onnx.load(onnx_fp32_path)
        input_name = onnx_model.graph.input[0].name
        calibration_data_reader = PlantVisionDataReader(
            dataloader=calib_dataloader,
            onnx_input_name=input_name,
        )

        # Perform static Quantization using the preprocessed model
        onnx_int8_path = output_dir / f"{base_filename}.int8.onnx"
        print(f"🔬 Calibrating and quantizing the model. Saving to: {onnx_int8_path}")
        quantize_static(
            model_input=preprocessed_fp32_path,
            model_output=onnx_int8_path,
            calibration_data_reader=calibration_data_reader,
            quant_format='QDQ',
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        print("☑️ INT8 Static Quantization complete!")
        print(f"\n Original FP32 size: {onnx_fp32_path.stat().st_size / 1e6:.2f} MB")
        print(f"\n Quantized INT8 size: {onnx_int8_path.stat().st_size / 1e6:.2f} MB")

    print("\n✅ Export Finished")


def main():
    parser = argparse.ArgumentParser(description="Export a trained PlantVision model to ONNX.")

    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="outputs/best_model.pth",
        help="Path to the trained PyTorch .pth file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exported_model/",
        help="Path to save the output ONNX model (extension will be added)",
    )
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization to the model",
    )

    args = parser.parse_args()

    model_path = paths.PROJECT_ROOT / args.model_checkpoint
    output_dir_path = paths.PROJECT_ROOT / args.output_dir

    export_model(
        model_checkpoint=model_path,
        output_dir=output_dir_path,
        quantize=args.quantize,
    )

if __name__ == "__main__":
    main()
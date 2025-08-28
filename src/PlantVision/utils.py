import yaml
import torch
from onnxruntime.quantization import CalibrationDataReader


# Helper function for opening .yaml configuration files
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Helper class for static quantization
class PlantVisionDataReader(CalibrationDataReader):
    def __init__(self, dataloader: torch.utils.data.DataLoader, onnx_input_name: str):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.onnx_input_name = onnx_input_name

    def get_next(self):
        try:
            images, _ = next(self.iterator)
            # ONNX Runtime expects a dictionary mapping input names to numpy arrays
            return {self.onnx_input_name: images.numpy()}
        except StopIteration:
            return None # Signal the end of calibration data
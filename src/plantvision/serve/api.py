import io
import json
import torch
import onnxruntime
from PIL import Image
from fastapi import FastAPI, UploadFile, File

from plantvision import paths
from torchvision import transforms
from plantvision.utils import load_config

DATA_CONFIG_PATH = paths.CONFIG_DIR / "data_config.yaml"
data_config = load_config(DATA_CONFIG_PATH)['data']

IMG_SIZE = data_config['img_size']
MEAN = data_config['mean']
STD = data_config['std']

# Transformation pipeline
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Load class names
CLASS_NAMES_PATH = paths.OUTPUTS_DIR / "class_names.json"
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Load the ONNX model and create an inference session
# Use the small, fast quantized model
MODEL_PATH = paths.OUTPUTS_DIR / "exported_model" / "plantvision_b0.int8.onnx"
SESSION = onnxruntime.InferenceSession(str(MODEL_PATH))
INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_NAME = SESSION.get_outputs()[0].name

# Create the FastAPI app inference
app = FastAPI(title="PlantVision CV001DD API", description="API for plant disease classification")

@app.get("/")
def read_root():
    return {"message": "Welcome to the PlantVision API. Use the /predict endpoint to classify a plant leaf image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, preprocesses it, runs inference and returns the prediction
    """
    # Read and process the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess the image
    input_tensor = TRANSFORMS(image).unsqueeze(0) # Add batch dim
    input_np = input_tensor.numpy()

    # Run inference with ONNX Runtime
    result = SESSION.run([OUTPUT_NAME], {INPUT_NAME: input_np})

    # Post-process the output: Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(result[0]), dim=1)

    # Get the top prediction
    top_prob, top_catid = torch.max(probabilities, 1)
    predicted_class = CLASS_NAMES[top_catid.item()]
    confidence = top_prob.item()

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
    }





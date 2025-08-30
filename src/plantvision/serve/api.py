import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms

IMG_SIZE = 224



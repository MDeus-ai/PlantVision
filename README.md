# 🌿 PlantVision

<p align="center">
  <img src="logo/logo.png" alt="PlantVision Banner" width="100%" height="600"/>
</p>

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-B2-009688?style=flat-square)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)


---

**PlantVision** is a deep learning–powered plant disease detection system built with Python and PyTorch. It leverages the EfficientNet-B2 architecture to classify 69 plant disease categories with high accuracy. Designed for offline mobile deployment, PlantVision aims to assist farmers, gardeners, and agricultural researchers in identifying plant diseases directly in the field.

---

## 📑 Table of Contents

| Section                                 | Link                                                               |
|-----------------------------------------|--------------------------------------------------------------------|
| 🚀 Features                             | [Jump to Features](#-features)                                     |
| 📂 Directory Structure                  | [Jump to Directory Structure](#-directory-structure)               |
| 🛠️ Installation                         | [Jump to Installation](#️-installation)                             |
| 📊 Dataset                              | [Jump to Dataset](#-dataset)                                       |
| 🏋️‍♂️ Training                          | [Jump to Training](#️-training)                                    |
| 📈 Evaluation                           | [Jump to Evaluation](#-evaluation)                                 |
| 🖼️ Inference                            | [Jump to Inference](#-inference)                                   |
| &nbsp;&nbsp;&nbsp;• Single-Image Inference | [Jump to Single-Image Inference](#single-image-inference)         |
| &nbsp;&nbsp;&nbsp;• Batch Inference       | [Jump to Batch Inference](#batch-inference)                       |
| 📱 Mobile Deployment                    | [Jump to Mobile Deployment](#-mobile-deployment)                   |
| &nbsp;&nbsp;&nbsp;• Export to TorchScript  | [Jump to Export to TorchScript](#export-to-torchscript)           |
| &nbsp;&nbsp;&nbsp;• Export to ONNX         | [Jump to Export to ONNX](#export-to-onnx)                         |
| &nbsp;&nbsp;&nbsp;• Integration            | [Jump to Integration](#integration)                               |
| 🤝 Contributing                         | [Jump to Contributing](#-contributing)                             |
| 📄 License                              | [Jump to License](#-license)                                       |
| 📬 Contact                              | [Jump to Contact](#-contact)                                       |

---

## 🚀 Features

- **State-of-the-art Backbone**  
  Utilizes EfficientNet-B2 for robust feature extraction.

- **Offline Inference**  
  Optimized for deployment on mobile devices without internet connectivity.

- **Mobile-Friendly Model Export**  
  Supports TorchScript and ONNX formats for seamless integration into mobile applications.

- **Comprehensive Pipeline**  
  Includes data preprocessing, augmentation, training, evaluation, and deployment modules.

- **Extensibility**  
  Modular design allows easy customization and extension for additional plant species or diseases.

---

## 📂 Directory Structure


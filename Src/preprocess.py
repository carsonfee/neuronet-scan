from PIL import Image
import numpy as np
import json

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and resize image, normalize pixel values."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return image_array

def preprocess_metadata(json_path):
    """Load and normalize patient metadata (e.g., for PyTorch models)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Example: manually extract and normalize known fields
    age = data.get("age", 0) / 100.0
    sex = 1.0 if data.get("sex", "M") == "M" else 0.0
    symptoms = data.get("symptoms", [])
    num_symptoms = len(symptoms) / 10.0  # crude normalization

    return np.array([age, sex, num_symptoms])
  

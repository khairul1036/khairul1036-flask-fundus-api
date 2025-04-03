import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained deep learning model
MODEL_PATH = "fundus_model_inceptionV3.h5"  # Change this to your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Update these based on your dataset)
CLASS_NAMES = ['Central Serous Chorioretinopathy', 'Diabetic Retinopathy', 'Disc Edema', 'Glaucoma', 'Healthy',
               'Macular Scar', 'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa']

# Image preprocessing function
def preprocess_image(image):
    """Preprocess the image: Convert to RGB, resize, normalize, and add batch dimension."""
    img = image.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size (change if needed)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/")
def home():
    return "<p>server is running....</p>"

# API route to handle image classification via URL
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data['image_url']

    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error if download fails

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        return jsonify({"predicted_class": predicted_class})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io

# Import the FungusClassifier class
from classifier.fungus_classifier import FungusClassifier

# Import OpenAI function
from openaiDataRetrieval import get_fungus_info_from_chatgpt  # Ensure this is the correct path

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend requests from Next.js

# Load model from the correct path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bay_area.pth")

# Initialize the classifier
classifier = FungusClassifier(model_path=MODEL_PATH, device="cpu")

@app.route("/classify", methods=["POST"])
def classify_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Convert the uploaded image to a format suitable for the classifier
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Get prediction
    predicted_species = classifier.predict(image)

    # Fetch additional information from OpenAI
    fungus_info = get_fungus_info_from_chatgpt(predicted_species)

    return jsonify({
        "species": predicted_species,
        "fungus_info": fungus_info  # Include the OpenAI response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

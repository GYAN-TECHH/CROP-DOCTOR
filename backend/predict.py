import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"

# -----------------------------
# Load trained model
# -----------------------------
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load class names
# -----------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Model loaded successfully!")
print(f"Number of classes: {len(class_names)}")


# -----------------------------
# Prediction function
# -----------------------------
def predict_disease(image_path):

    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    #
    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The trained model already contains:
    # layers.Rescaling(1.0 / 255)
    #
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    return predicted_class, confidence


# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":

    print("\n🌱 AI Crop Doctor")
    print("========================")

    image_path = input(
        "Enter image path: "
    ).strip()

    # Remove quotes if user pasted
    # a path like "C:\image.jpg"
    image_path = image_path.strip('"').strip("'")

    try:

        disease, confidence = predict_disease(
            image_path
        )

        print("\n🌱 Prediction Result")
        print("========================")

        print(
            f"Disease: {disease}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

    except FileNotFoundError as e:

        print("\n❌ ERROR:")
        print(e)

    except Exception as e:

        print("\n❌ Something went wrong:")
        print(e)
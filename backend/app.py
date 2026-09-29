from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from PIL import Image
import numpy as np

import os
import uuid

# =========================================================
# GEMINI
# =========================================================

from google import genai


# =========================================================
# 1. FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.txt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# 3. IMAGE SETTINGS
# =========================================================

IMG_SIZE = 224


# =========================================================
# 4. LOAD CNN MODEL
# =========================================================

print()
print("AI Crop Doctor")
print("==============================")
print("Loading CNN model...")

model = None

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("CNN model loaded successfully!")

except Exception as e:

    print("ERROR: Could not load model.")
    print(str(e))


# =========================================================
# 5. LOAD CLASS NAMES
# =========================================================

class_names = []

print("Loading class names...")

try:

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        "Number of classes:",
        len(class_names)
    )

    print("Classes:")

    for i, name in enumerate(class_names):

        print(
            i,
            "->",
            name
        )

except Exception as e:

    print("ERROR: Could not load class names.")
    print(str(e))


# =========================================================
# 6. GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini configured successfully!"
        )

    except Exception as e:

        print(
            "Gemini configuration error:"
        )

        print(str(e))

else:

    print(
        "WARNING: GEMINI_API_KEY not found."
    )


# =========================================================
# 7. GEMINI MODEL
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"


# =========================================================
# 8. CHECK MODEL
# =========================================================

if model is not None:

    try:

        output_shape = model.output_shape

        print(
            "CNN output shape:",
            output_shape
        )

    except Exception:

        pass


# =========================================================
# 9. CNN PREDICTION FUNCTION
# =========================================================

def predict_disease(image_path):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        raise Exception(
            "CNN model is not loaded."
        )

    # -----------------------------------------------------
    # Check class names
    # -----------------------------------------------------

    if len(class_names) == 0:

        raise Exception(
            "Class names are not loaded."
        )

    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # -----------------------------------------------------
    # Resize
    # -----------------------------------------------------

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # -----------------------------------------------------
    # Convert to numpy
    # -----------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    #
    # Do NOT divide by 255 here.
    #
    # Your CNN model already has:
    #
    # layers.Rescaling(1.0 / 255)
    #
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print()
    print("Image shape:")
    print(image_array.shape)

    # -----------------------------------------------------
    # CNN prediction
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    # -----------------------------------------------------
    # Prediction values
    # -----------------------------------------------------

    prediction_values = predictions[0]

    # -----------------------------------------------------
    # Predicted index
    # -----------------------------------------------------

    predicted_index = int(
        np.argmax(
            prediction_values
        )
    )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = float(
        prediction_values[
            predicted_index
        ]
    ) * 100

    # -----------------------------------------------------
    # Class name
    # -----------------------------------------------------

    if predicted_index < len(
        class_names
    ):

        predicted_class = (
            class_names[
                predicted_index
            ]
        )

    else:

        predicted_class = "Unknown"

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print()
    print("CNN Prediction Result")
    print("==============================")

    print(
        "Disease:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print(
        "Class Index:",
        predicted_index
    )

    print("==============================")
    print()

    return (
        predicted_class,
        confidence
    )


# =========================================================
# 10. GEMINI ADVICE FUNCTION
# =========================================================

def get_gemini_advice(
    disease,
    confidence
):

    # -----------------------------------------------------
    # Check Gemini
    # -----------------------------------------------------

    if gemini_client is None:

        return {
            "status": "error",
            "message":
                "Gemini API is not configured."
        }

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""

You are an AI Crop Doctor assistant.

A CNN image classification model detected:

Disease: {disease}
CNN Confidence: {confidence:.2f}%

Provide useful information for a farmer.

Give the answer in simple English.

Include:

1. Disease name
2. What this disease means
3. Main symptoms
4. Possible cause
5. Treatment
6. Prevention
7. Basic farming advice

Important:
- Do not say the diagnosis is 100% certain.
- The CNN prediction is an AI prediction.
- If the confidence is low, advise the farmer to consult an agricultural expert.
- Do not recommend unsafe chemical use.
- Keep the response practical and easy to understand.

"""

    # -----------------------------------------------------
    # Call Gemini
    # -----------------------------------------------------

    try:

        response = gemini_client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt

        )

        return {

            "status": "success",

            "advice":
                response.text

        }

    except Exception as e:

        print()
        print("Gemini Error:")
        print(str(e))

        return {

            "status": "error",

            "message":
                str(e)

        }


# =========================================================
# 11. HOME ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "success",

        "message":
            "AI Crop Doctor API is running",

        "classes":
            len(class_names),

        "gemini":
            gemini_client is not None

    })


# =========================================================
# 12. GEMINI TEST ROUTE
# =========================================================

@app.route(
    "/gemini-test",
    methods=["GET"]
)
def gemini_test():

    if gemini_client is None:

        return jsonify({

            "status": "error",

            "error":
                "GEMINI_API_KEY is not configured."

        }), 500

    try:

        response = gemini_client.models.generate_content(

            model=GEMINI_MODEL,

            contents=
                "Explain plant disease in one simple sentence."

        )

        return jsonify({

            "status": "success",

            "response":
                response.text

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "error":
                str(e)

        }), 500


# =========================================================
# 13. PREDICTION ROUTE
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    image_path = None

    try:

        # -------------------------------------------------
        # Check image
        # -------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "status": "error",

                "error":
                    "No image uploaded."

            }), 400

        # -------------------------------------------------
        # Get file
        # -------------------------------------------------

        file = request.files[
            "image"
        ]

        # -------------------------------------------------
        # Check filename
        # -------------------------------------------------

        if file.filename == "":

            return jsonify({

                "status": "error",

                "error":
                    "No file selected."

            }), 400

        # -------------------------------------------------
        # Allowed extensions
        # -------------------------------------------------

        allowed_extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".webp"

        }

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        if extension not in allowed_extensions:

            return jsonify({

                "status": "error",

                "error":
                    "Invalid image format. "
                    "Use JPG, JPEG, PNG or WEBP."

            }), 400

        # -------------------------------------------------
        # Unique filename
        # -------------------------------------------------

        unique_filename = (

            str(uuid.uuid4())

            + extension

        )

        image_path = os.path.join(

            UPLOAD_FOLDER,

            unique_filename

        )

        # -------------------------------------------------
        # Save image
        # -------------------------------------------------

        file.save(
            image_path
        )

        print()
        print("New image received:")
        print(file.filename)

        # -------------------------------------------------
        # CNN PREDICTION
        # -------------------------------------------------

        disease, confidence = (
            predict_disease(
                image_path
            )
        )

        # -------------------------------------------------
        # GEMINI ADVICE
        # -------------------------------------------------

        gemini_result = get_gemini_advice(

            disease,

            confidence

        )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "disease":
                disease,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "gemini":
                gemini_result

        })

    except Exception as e:

        print()
        print("Prediction error:")
        print(str(e))

        return jsonify({

            "status": "error",

            "error":
                str(e)

        }), 500

    finally:

        # -------------------------------------------------
        # Delete uploaded image
        # -------------------------------------------------

        if image_path is not None:

            try:

                if os.path.exists(
                    image_path
                ):

                    os.remove(
                        image_path
                    )

            except Exception:

                pass


# =========================================================
# 14. RUN SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "AI Crop Doctor Backend Starting..."
    )

    print(
        "=================================="
    )

    print(
        "CNN Model:",
        model is not None
    )

    print(
        "Classes:",
        len(class_names)
    )

    print(
        "Gemini:",
        gemini_client is not None
    )

    print()

    print(
        "Server:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()
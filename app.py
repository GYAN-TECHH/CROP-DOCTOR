# =========================================================
# AI CROP DOCTOR
# Flask + CNN + Gemini AI
# =========================================================

import os
import re
import time

import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

from tensorflow.keras.models import load_model
from google import genai


# =========================================================
# 1. BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# 2. LOAD ENVIRONMENT
# =========================================================

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# =========================================================
# 3. FLASK APP
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# 4. UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# 5. MODEL PATH
# =========================================================

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


# =========================================================
# 6. GLOBAL VARIABLES
# =========================================================

cnn_model = None

class_names = []

gemini_client = None


# =========================================================
# 7. GEMINI MODELS
# =========================================================

GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]


# =========================================================
# 8. STARTING MESSAGE
# =========================================================

print()
print("==============================================")
print("          AI CROP DOCTOR - STARTING")
print("==============================================")


# =========================================================
# 9. LOAD CNN MODEL
# =========================================================

print()
print("Loading CNN model...")

print(
    "Model path:",
    MODEL_PATH
)

print(
    "Model exists:",
    os.path.isfile(MODEL_PATH)
)


try:

    if os.path.isfile(MODEL_PATH):

        cnn_model = load_model(
            MODEL_PATH,
            compile=False
        )

        print()
        print("CNN MODEL LOADED SUCCESSFULLY")

        print(
            "Input shape:",
            cnn_model.input_shape
        )

        print(
            "Output shape:",
            cnn_model.output_shape
        )

    else:

        print()
        print("CNN MODEL NOT FOUND")

except Exception as e:

    cnn_model = None

    print()
    print("CNN MODEL LOADING ERROR")

    print(
        type(e).__name__
    )

    print(
        str(e)
    )


# =========================================================
# 10. LOAD CLASS NAMES
# =========================================================

print()
print("Loading class names...")


try:

    if os.path.isfile(
        CLASS_NAMES_PATH
    ):

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            class_names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print()
        print(
            "CLASS NAMES LOADED:",
            len(class_names)
        )

        for i, name in enumerate(
            class_names
        ):

            print(
                f"{i} -> {name}"
            )

    else:

        print()
        print("class_names.txt NOT FOUND")

except Exception as e:

    class_names = []

    print()
    print("CLASS NAME ERROR")

    print(
        str(e)
    )


# =========================================================
# 11. GEMINI SETUP
# =========================================================

print()
print("Loading Gemini AI...")


if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print()
        print("GEMINI API READY")

    except Exception as e:

        gemini_client = None

        print()
        print("GEMINI INITIALIZATION ERROR")

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

else:

    print()
    print("GEMINI_API_KEY NOT FOUND")

    print(
        "Check your .env file."
    )


# =========================================================
# 12. GET MODEL IMAGE SIZE
# =========================================================

def get_model_size():

    if cnn_model is None:

        return 224, 224

    try:

        input_shape = (
            cnn_model.input_shape
        )

        height = input_shape[1]
        width = input_shape[2]

        if height is None:

            height = 224

        if width is None:

            width = 224

        return (
            int(width),
            int(height)
        )

    except Exception:

        return 224, 224


# =========================================================
# 13. IMAGE PREPROCESSING
# =========================================================

def prepare_image(image):

    width, height = (
        get_model_size()
    )

    image = image.resize(
        (width, height)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Check if model already has
    # a Rescaling layer.

    try:

        has_rescaling = any(
            layer.__class__.__name__
            == "Rescaling"
            for layer in cnn_model.layers
        )

    except Exception:

        has_rescaling = False


    if not has_rescaling:

        image_array = (
            image_array / 255.0
        )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# 14. FORMAT DISEASE NAME
# =========================================================

def format_disease_name(name):

    name = str(name)

    name = name.replace(
        "___",
        " - "
    )

    name = name.replace(
        "_",
        " "
    )

    return name.strip()


# =========================================================
# 15. EXTRACT GEMINI SECTION
# =========================================================

def extract_section(
    text,
    section_name,
    next_sections
):

    if not text:

        return ""


    escaped_name = re.escape(
        section_name
    )


    if next_sections:

        next_pattern = "|".join(
            re.escape(section)
            for section in next_sections
        )

        pattern = (
            rf"\**{escaped_name}\**\s*:?\s*"
            rf"(.*?)"
            rf"(?=\n\s*\**"
            rf"(?:{next_pattern})"
            rf"\**\s*:|\Z)"
        )

    else:

        pattern = (
            rf"\**{escaped_name}\**\s*:?\s*"
            rf"(.*?)"
            rf"\Z"
        )


    match = re.search(
        pattern,
        text,
        flags=
        re.IGNORECASE |
        re.DOTALL
    )


    if not match:

        return ""


    return match.group(1).strip()


# =========================================================
# 16. GENERATE GEMINI ADVICE
# =========================================================

def generate_gemini_advice(disease):

    print()
    print("==============================================")
    print("              GEMINI AI REQUEST")
    print("==============================================")

    print(
        "Disease:",
        disease
    )


    # -----------------------------------------------------
    # Check Gemini client
    # -----------------------------------------------------

    if gemini_client is None:

        return {

            "cause": "",

            "symptoms": "",

            "treatment": "",

            "prevention": "",

            "raw_advice": "",

            "error":
                "Gemini API is not configured."

        }


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are an agricultural AI assistant.

Detected plant disease:
{disease}

Provide practical and simple advice for a farmer.

Return EXACTLY this format:

Cause:
Write 2 short sentences.

Symptoms:
- symptom 1
- symptom 2
- symptom 3

Treatment:
- treatment step 1
- treatment step 2
- treatment step 3
- treatment step 4

Prevention:
- prevention step 1
- prevention step 2
- prevention step 3
- prevention step 4

Important:
Do not invent pesticide brand names.
For pesticides or fungicides, tell the user to follow the product label
and local agricultural recommendations.

Keep the advice practical and concise.
"""


    # -----------------------------------------------------
    # Try models in order
    # -----------------------------------------------------

    last_error = ""


    for model_name in GEMINI_MODELS:

        print()
        print(
            "Trying Gemini model:",
            model_name
        )


        # -------------------------------------------------
        # Retry each model 3 times
        # -------------------------------------------------

        for attempt in range(3):

            try:

                response = (
                    gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                )


                advice_text = (
                    response.text
                    if response.text
                    else ""
                )


                if not advice_text.strip():

                    raise Exception(
                        "Gemini returned an empty response."
                    )


                print()
                print(
                    "GEMINI RESPONSE RECEIVED"
                )

                print(
                    "Model:",
                    model_name
                )

                print()
                print(
                    advice_text
                )


                # -------------------------------------------------
                # Extract sections
                # -------------------------------------------------

                cause = extract_section(
                    advice_text,
                    "Cause",
                    [
                        "Symptoms",
                        "Treatment",
                        "Prevention"
                    ]
                )


                symptoms = extract_section(
                    advice_text,
                    "Symptoms",
                    [
                        "Treatment",
                        "Prevention"
                    ]
                )


                treatment = extract_section(
                    advice_text,
                    "Treatment",
                    [
                        "Prevention"
                    ]
                )


                prevention = extract_section(
                    advice_text,
                    "Prevention",
                    []
                )


                # -------------------------------------------------
                # Return successful response
                # -------------------------------------------------

                return {

                    "cause":
                        cause,

                    "symptoms":
                        symptoms,

                    "treatment":
                        treatment,

                    "prevention":
                        prevention,

                    "raw_advice":
                        advice_text,

                    "error":
                        ""

                }


            except Exception as e:

                last_error = str(e)


                print()
                print(
                    f"Gemini attempt "
                    f"{attempt + 1}/3 failed"
                )

                print(
                    "Model:",
                    model_name
                )

                print(
                    type(e).__name__
                )

                print(
                    str(e)
                )


                # -------------------------------------------------
                # Wait before retry
                # -------------------------------------------------

                if attempt < 2:

                    wait_time = (
                        2 ** attempt
                    )

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )


        # -------------------------------------------------
        # Current model failed
        # -------------------------------------------------

        print()
        print(
            "Model failed:",
            model_name
        )

        print(
            "Trying next model..."
        )


    # =====================================================
    # ALL MODELS FAILED
    # =====================================================

    print()
    print("==============================================")
    print("        ALL GEMINI MODELS FAILED")
    print("==============================================")


    return {

        "cause": "",

        "symptoms": "",

        "treatment": "",

        "prevention": "",

        "raw_advice": "",

        "error":
            last_error

    }


# =========================================================
# 17. HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status":
            "success",

        "message":
            "AI Crop Doctor API is running",

        "cnn_model":
            cnn_model is not None,

        "classes":
            len(class_names),

        "gemini":
            gemini_client is not None

    })


# =========================================================
# 18. HEALTH CHECK
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "success",

        "cnn_model":
            cnn_model is not None,

        "classes":
            len(class_names),

        "gemini":
            gemini_client is not None

    })


# =========================================================
# 19. PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # Check CNN
        # -------------------------------------------------

        if cnn_model is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    "CNN model is not loaded."

            }), 500


        # -------------------------------------------------
        # Check class names
        # -------------------------------------------------

        if not class_names:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Class names are not loaded."

            }), 500


        # -------------------------------------------------
        # Check image
        # -------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "status":
                    "error",

                "message":
                    "No image uploaded."

            }), 400


        file = request.files[
            "image"
        ]


        if file.filename == "":

            return jsonify({

                "status":
                    "error",

                "message":
                    "No image selected."

            }), 400


        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image = Image.open(
            file
        ).convert("RGB")


        # -------------------------------------------------
        # Prepare image
        # -------------------------------------------------

        image_array = (
            prepare_image(
                image
            )
        )


        print()
        print(
            "Image shape:",
            image_array.shape
        )


        # -------------------------------------------------
        # CNN prediction
        # -------------------------------------------------

        prediction = (
            cnn_model.predict(
                image_array,
                verbose=0
            )
        )


        prediction_values = np.array(
            prediction[0],
            dtype=np.float32
        )


        # -------------------------------------------------
        # Validate model output
        # -------------------------------------------------

        if len(prediction_values) != len(
            class_names
        ):

            return jsonify({

                "status":
                    "error",

                "message":
                    "Model output classes do not match class_names.txt.",

                "model_classes":
                    len(prediction_values),

                "class_names":
                    len(class_names)

            }), 500


        # -------------------------------------------------
        # Convert to probabilities
        # -------------------------------------------------

        total = float(
            np.sum(
                prediction_values
            )
        )


        if (
            np.any(
                prediction_values < 0
            )
            or
            not np.isclose(
                total,
                1.0,
                atol=0.05
            )
        ):

            shifted = (
                prediction_values
                -
                np.max(
                    prediction_values
                )
            )


            exp_values = np.exp(
                shifted
            )


            probabilities = (
                exp_values
                /
                np.sum(
                    exp_values
                )
            )

        else:

            probabilities = (
                prediction_values
            )


        # -------------------------------------------------
        # Top prediction
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        predicted_class = (
            class_names[
                predicted_index
            ]
        )


        disease_display = (
            format_disease_name(
                predicted_class
            )
        )


        # -------------------------------------------------
        # Top 3 predictions
        # -------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]


        top_predictions = []


        for index in top_indices:

            index = int(
                index
            )


            top_predictions.append({

                "disease":
                    format_disease_name(
                        class_names[index]
                    ),

                "confidence":
                    round(
                        float(
                            probabilities[index]
                        ) * 100,
                        2
                    )

            })


        # -------------------------------------------------
        # Print CNN result
        # -------------------------------------------------

        print()
        print("==============================================")
        print("             CNN PREDICTION")
        print("==============================================")


        print(
            "Disease:",
            disease_display
        )


        print(
            "Confidence:",
            round(
                confidence * 100,
                2
            ),
            "%"
        )


        print(
            "Class index:",
            predicted_index
        )


        print("==============================================")


        # -------------------------------------------------
        # Generate Gemini advice
        # -------------------------------------------------

        gemini_data = (
            generate_gemini_advice(
                disease_display
            )
        )


        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        return jsonify({

            "status":
                "success",

            "disease":
                disease_display,

            "prediction":
                disease_display,

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "class_index":
                predicted_index,

            "top_predictions":
                top_predictions,

            "cause":
                gemini_data["cause"],

            "symptoms":
                gemini_data["symptoms"],

            "treatment":
                gemini_data["treatment"],

            "prevention":
                gemini_data["prevention"],

            "advice":
                gemini_data["raw_advice"],

            "gemini_error":
                gemini_data["error"]

        })


    except Exception as e:

        print()
        print("==============================================")
        print("             PREDICTION ERROR")
        print("==============================================")


        print(
            type(e).__name__
        )


        print(
            str(e)
        )


        print("==============================================")


        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 20. DIRECT GEMINI ADVICE API
# =========================================================

@app.route(
    "/advice",
    methods=["POST"]
)
def advice():

    try:

        # -------------------------------------------------
        # Read JSON
        # -------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Invalid JSON request."

            }), 400


        # -------------------------------------------------
        # Get disease
        # -------------------------------------------------

        disease = data.get(
            "disease"
        )


        if not disease:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Disease field is required."

            }), 400


        disease = str(
            disease
        ).strip()


        # -------------------------------------------------
        # Print request
        # -------------------------------------------------

        print()
        print("==============================================")
        print("             DIRECT ADVICE API")
        print("==============================================")


        print(
            "Disease:",
            disease
        )


        # -------------------------------------------------
        # Generate Gemini advice
        # -------------------------------------------------

        gemini_data = (
            generate_gemini_advice(
                disease
            )
        )


        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return jsonify({

            "status":
                "success",

            "disease":
                disease,

            "cause":
                gemini_data["cause"],

            "symptoms":
                gemini_data["symptoms"],

            "treatment":
                gemini_data["treatment"],

            "prevention":
                gemini_data["prevention"],

            "advice":
                gemini_data["raw_advice"],

            "gemini_error":
                gemini_data["error"]

        })


    except Exception as e:

        print()
        print("==============================================")
        print("             ADVICE API ERROR")
        print("==============================================")


        print(
            type(e).__name__
        )


        print(
            str(e)
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# 21. RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("       AI CROP DOCTOR SERVER STARTED")
    print("==============================================")


    print(
        "URL: http://127.0.0.1:5000"
    )


    print(
        "Health: http://127.0.0.1:5000/health"
    )


    print(
        "Predict: http://127.0.0.1:5000/predict"
    )


    print(
        "Advice: http://127.0.0.1:5000/advice"
    )


    print("==============================================")


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
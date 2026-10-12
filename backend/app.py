# ============================================================
# AI CROP DOCTOR - FLASK BACKEND
# CNN Disease Detection + Gemini AI + Hugging Face Fallback
# ============================================================

import os
import json
import re
import traceback

import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Gemini
from google import genai

# Hugging Face
from huggingface_hub import InferenceClient


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)

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

ENV_PATH = os.path.join(
    BACKEND_DIR,
    ".env"
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

print("\n==========================================")
print("        AI CROP DOCTOR BACKEND")
print("==========================================")

print("\nLoading environment variables...")

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if GEMINI_API_KEY:
    print("✅ GEMINI_API_KEY loaded")
else:
    print("⚠️ GEMINI_API_KEY not found")

if HF_TOKEN:
    print("✅ HF_TOKEN loaded")
else:
    print("⚠️ HF_TOKEN not found")


# ============================================================
# 3. CREATE FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# 4. LOAD CNN MODEL
# ============================================================

print("\nLoading CNN model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("✅ CNN model loaded successfully")

except Exception as e:

    print("❌ Failed to load CNN model")
    print(e)

    model = None


# ============================================================
# 5. LOAD CLASS NAMES
# ============================================================

print("\nLoading class names...")

try:

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    print(
        f"✅ {len(class_names)} classes loaded"
    )

except Exception as e:

    print("❌ Failed to load class names")
    print(e)

    class_names = []


# ============================================================
# 6. CHECK MODEL RESCALING
# ============================================================

HAS_RESCALING = False

if model is not None:

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Rescaling
        ):

            HAS_RESCALING = True
            break


if HAS_RESCALING:

    print(
        "✅ Model contains Rescaling layer"
    )

    print(
        "➡️ Backend will NOT divide image by 255"
    )

else:

    print(
        "⚠️ No Rescaling layer found"
    )

    print(
        "➡️ Backend will divide image by 255"
    )


# ============================================================
# 7. GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "✅ Gemini client initialized"
        )

    except Exception as e:

        print(
            "❌ Gemini initialization failed:"
        )

        print(e)


# ============================================================
# 8. GEMINI MODEL FALLBACK LIST
# ============================================================

# We try multiple models because temporary service
# overload can happen.

GEMINI_MODELS = [

    "gemini-3.8-flash",

    "gemini-3.6-flash",

    "gemini-3.5-flash-lite",

    "gemini-3.5-flash",

    "gemini-3.1-flash-lite"

]


# ============================================================
# 9. HUGGING FACE CLIENT
# ============================================================

hf_client = None

if HF_TOKEN:

    try:

        hf_client = InferenceClient(
            api_key=HF_TOKEN,
            provider="auto"
        )

        print(
            "✅ Hugging Face client initialized"
        )

    except Exception as e:

        print(
            "❌ Hugging Face initialization failed:"
        )

        print(e)


# ============================================================
# 10. ALLOWED IMAGE EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {

    "jpg",
    "jpeg",
    "png",
    "webp"

}


def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS

    )


# ============================================================
# 11. CLEAN DISEASE NAME
# ============================================================

def clean_disease_name(name):

    name = name.replace(
        "___",
        " "
    )

    name = name.replace(
        "__",
        " "
    )

    name = name.replace(
        "_",
        " "
    )

    return name.strip()


# ============================================================
# 12. FORMAT LIST INTO TEXT
# ============================================================

def format_list(value):

    if value is None:

        return ""

    if isinstance(
        value,
        list
    ):

        return "\n".join(
            [
                f"• {str(item).strip()}"
                for item in value
                if str(item).strip()
            ]
        )

    return str(value).strip()


# ============================================================
# 13. CLEAN GEMINI / HF RESPONSE
# ============================================================

def clean_json_response(text):

    if not text:

        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # Try direct JSON
    try:

        return json.loads(text)

    except Exception:

        pass

    # Try extracting JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:

            pass

    return None


# ============================================================
# 14. CREATE AI PROMPT
# ============================================================

def create_ai_prompt(disease):

    prompt = f"""
You are an agricultural plant disease assistant.

The CNN model detected this plant disease:

Disease:
{disease}

Provide practical and easy-to-understand information for a farmer.

Return ONLY valid JSON.

Do not use Markdown.
Do not use code fences.
Do not add any explanation outside JSON.

Use exactly this structure:

{{
    "cause": "Short explanation of the main cause of the disease.",
    "symptoms": [
        "Symptom 1",
        "Symptom 2",
        "Symptom 3"
    ],
    "treatment": [
        "Treatment recommendation 1",
        "Treatment recommendation 2",
        "Treatment recommendation 3"
    ],
    "prevention": [
        "Prevention recommendation 1",
        "Prevention recommendation 2",
        "Prevention recommendation 3"
    ]
}}

Important:
- Keep the advice practical.
- Avoid extremely technical language.
- Do not invent unrelated diseases.
- If chemical treatment is mentioned, advise following the product label and local agricultural guidance.
"""

    return prompt


# ============================================================
# 15. NORMALIZE AI RESULT
# ============================================================

def normalize_ai_result(data):

    if not isinstance(
        data,
        dict
    ):

        return None

    return {

        "cause": format_list(
            data.get(
                "cause",
                ""
            )
        ),

        "symptoms": format_list(
            data.get(
                "symptoms",
                []
            )
        ),

        "treatment": format_list(
            data.get(
                "treatment",
                []
            )
        ),

        "prevention": format_list(
            data.get(
                "prevention",
                []
            )
        )

    }


# ============================================================
# 16. GEMINI AI FUNCTION
# ============================================================

def get_gemini_advice(disease):

    if not gemini_client:

        raise Exception(
            "Gemini client is not available"
        )

    prompt = create_ai_prompt(
        disease
    )

    errors = []

    print("\n------------------------------------------")
    print("Trying Gemini AI...")
    print("------------------------------------------")

    for model_name in GEMINI_MODELS:

        try:

            print(
                f"➡️ Trying model: {model_name}"
            )

            response = gemini_client.models.generate_content(

                model=model_name,

                contents=prompt

            )

            response_text = response.text

            print(
                f"✅ Gemini response received from {model_name}"
            )

            data = clean_json_response(
                response_text
            )

            if data is None:

                raise Exception(
                    "Gemini returned invalid JSON"
                )

            result = normalize_ai_result(
                data
            )

            if result is None:

                raise Exception(
                    "Unable to normalize Gemini response"
                )

            result["provider"] = "Gemini"

            result["model"] = model_name

            return result

        except Exception as e:

            error_message = str(e)

            print(
                f"❌ {model_name} failed:"
            )

            print(error_message)

            errors.append(
                f"{model_name}: {error_message}"
            )

    raise Exception(
        "All Gemini models failed:\n"
        + "\n".join(errors)
    )


# ============================================================
# 17. HUGGING FACE AI FUNCTION
# ============================================================

def get_huggingface_advice(disease):

    if not hf_client:

        raise Exception(
            "Hugging Face client is not available"
        )

    print("\n------------------------------------------")
    print("Trying Hugging Face AI...")
    print("------------------------------------------")

    prompt = create_ai_prompt(
        disease
    )

    model_name = "openai/gpt-oss-120b"

    try:

        print(
            f"➡️ Hugging Face model: {model_name}"
        )

        response = hf_client.chat.completions.create(

            model=model_name,

            messages=[

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.2,

            max_tokens=700

        )

        response_text = (
            response
            .choices[0]
            .message
            .content
        )

        print(
            "✅ Hugging Face response received"
        )

        data = clean_json_response(
            response_text
        )

        if data is None:

            raise Exception(
                "Hugging Face returned invalid JSON"
            )

        result = normalize_ai_result(
            data
        )

        if result is None:

            raise Exception(
                "Unable to normalize Hugging Face response"
            )

        result["provider"] = "Hugging Face"

        result["model"] = model_name

        return result

    except Exception as e:

        print(
            "❌ Hugging Face failed:"
        )

        print(e)

        raise


# ============================================================
# 18. MAIN AI FUNCTION
# ============================================================

def get_ai_advice(disease):

    # ------------------------------------------
    # FIRST: GEMINI
    # ------------------------------------------

    if gemini_client:

        try:

            result = get_gemini_advice(
                disease
            )

            return result, None

        except Exception as gemini_error:

            print(
                "\n⚠️ Gemini failed."
            )

            print(
                gemini_error
            )

    # ------------------------------------------
    # SECOND: HUGGING FACE
    # ------------------------------------------

    if hf_client:

        try:

            result = get_huggingface_advice(
                disease
            )

            print(
                "\n✅ Hugging Face fallback successful"
            )

            return result, None

        except Exception as hf_error:

            print(
                "\n❌ Hugging Face fallback failed."
            )

            print(
                hf_error
            )

            return (

                None,

                "Gemini and Hugging Face AI services "
                "are currently unavailable."

            )

    # ------------------------------------------
    # NO AI PROVIDER
    # ------------------------------------------

    return (

        None,

        "No AI provider is configured. "
        "Please check GEMINI_API_KEY or HF_TOKEN."

    )


# ============================================================
# 19. CNN PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    if model is None:

        raise Exception(
            "CNN model is not loaded."
        )

    if not class_names:

        raise Exception(
            "Class names are not loaded."
        )

    # Open image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize
    image = image.resize(
        (224, 224)
    )

    # Convert to numpy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # If model already contains Rescaling(1/255),
    # DO NOT divide here.
    # --------------------------------------------------------

    if not HAS_RESCALING:

        image_array = (
            image_array / 255.0
        )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # CNN prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Top prediction
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({

            "disease": clean_disease_name(
                class_names[index]
            ),

            "raw_class": class_names[index],

            "confidence": round(
                float(
                    predictions[index]
                ) * 100,
                2
            )

        })

    return (

        predicted_class,

        confidence,

        top_predictions

    )


# ============================================================
# 20. HEALTH CHECK ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "success",

        "message": "🌱 AI Crop Doctor API is running",

        "cnn_model": (
            "loaded"
            if model is not None
            else "not loaded"
        ),

        "classes": len(
            class_names
        ),

        "gemini": (
            "configured"
            if gemini_client
            else "not configured"
        ),

        "huggingface": (
            "configured"
            if hf_client
            else "not configured"
        )

    })


# ============================================================
# 21. PREDICT ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ------------------------------------------
        # CHECK FILE
        # ------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "status": "error",

                "message":
                "No image file received."

            }), 400

        file = request.files[
            "image"
        ]

        if file.filename == "":

            return jsonify({

                "status": "error",

                "message":
                "No image selected."

            }), 400

        # ------------------------------------------
        # CHECK EXTENSION
        # ------------------------------------------

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "status": "error",

                "message":
                "Invalid image format. "
                "Use JPG, JPEG, PNG or WEBP."

            }), 400

        # ------------------------------------------
        # SAVE IMAGE
        # ------------------------------------------

        filename = secure_filename(
            file.filename
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(
            image_path
        )

        print("\n==========================================")
        print("NEW PREDICTION REQUEST")
        print("==========================================")

        print(
            f"Image: {filename}"
        )

        # ------------------------------------------
        # CNN PREDICTION
        # ------------------------------------------

        disease, confidence, top_predictions = (
            predict_image(
                image_path
            )
        )

        clean_name = clean_disease_name(
            disease
        )

        print(
            f"\n🌱 Disease: {clean_name}"
        )

        print(
            f"📊 Confidence: {confidence:.2f}%"
        )

        print("\nTop predictions:")

        for item in top_predictions:

            print(
                f"  {item['disease']} "
                f"→ {item['confidence']}%"
            )

        # ------------------------------------------
        # AI ADVICE
        # ------------------------------------------

        ai_result, ai_error = get_ai_advice(
            clean_name
        )

        # ------------------------------------------
        # DEFAULT AI VALUES
        # ------------------------------------------

        if ai_result:

            cause = ai_result.get(
                "cause",
                ""
            )

            symptoms = ai_result.get(
                "symptoms",
                ""
            )

            treatment = ai_result.get(
                "treatment",
                ""
            )

            prevention = ai_result.get(
                "prevention",
                ""
            )

            provider = ai_result.get(
                "provider",
                ""
            )

            ai_model = ai_result.get(
                "model",
                ""
            )

        else:

            cause = ""

            symptoms = ""

            treatment = ""

            prevention = ""

            provider = ""

            ai_model = ""

        # ------------------------------------------
        # FINAL RESPONSE
        # ------------------------------------------

        response_data = {

            "status": "success",

            "disease": clean_name,

            "raw_disease": disease,

            "confidence": round(
                confidence,
                2
            ),

            "top_predictions":
            top_predictions,

            # Gemini / LLM information
            "cause": cause,

            "symptoms": symptoms,

            "treatment": treatment,

            "prevention": prevention,

            "ai_provider": provider,

            "ai_model": ai_model,

            "gemini_error":
                ai_error

        }

        print("\n==========================================")
        print("RESPONSE READY")
        print("==========================================")

        if provider:

            print(
                f"🤖 AI Provider: {provider}"
            )

            print(
                f"🧠 AI Model: {ai_model}"
            )

        elif ai_error:

            print(
                "⚠️ AI advice unavailable"
            )

            print(
                ai_error
            )

        return jsonify(
            response_data
        )


    except Exception as e:

        print("\n==========================================")
        print("❌ PREDICTION ERROR")
        print("==========================================")

        print(
            str(e)
        )

        traceback.print_exc()

        return jsonify({

            "status": "error",

            "message":
            str(e)

        }), 500


# ============================================================
# 22. RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n==========================================")
    print("🌱 AI CROP DOCTOR SERVER")
    print("==========================================")

    print(
        f"📁 Model: {MODEL_PATH}"
    )

    print(
        f"📁 Classes: {CLASS_NAMES_PATH}"
    )

    print(
        f"📁 Uploads: {UPLOAD_FOLDER}"
    )

    print(
        "\n🌐 Backend URL:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\n🔗 Prediction API:"
    )

    print(
        "http://127.0.0.1:5000/predict"
    )

    print(
        "\n=========================================="
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
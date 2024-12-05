import os
import jwt
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import mysql.connector
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()


MODEL_SEVERITY_PATH = "model_frozen.tflite"  # Model untuk keparahan
MODEL_DISEASE_PATH = "model_jenis_penyakit_frozen.tflite"  # Model untuk penyakit kulit

# Load severity model
try:
    interpreter_severity = tf.lite.Interpreter(model_path=MODEL_SEVERITY_PATH)
    interpreter_severity.allocate_tensors()
    input_details_severity = interpreter_severity.get_input_details()
    output_details_severity = interpreter_severity.get_output_details()
except Exception as e:
    raise ValueError(f"Gagal memuat model keparahan: {str(e)}")

# Load disease identification model
try:
    interpreter_disease = tf.lite.Interpreter(model_path=MODEL_DISEASE_PATH)
    interpreter_disease.allocate_tensors()
    input_details_disease = interpreter_disease.get_input_details()
    output_details_disease = interpreter_disease.get_output_details()
except Exception as e:
    raise ValueError(f"Gagal memuat model penyakit: {str(e)}")

# Database connection
db_config = {
    "host": "34.34.218.3",
    "user": "root",
    "password": "lensa",
    "database": "auth_db"
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        raise Exception(f"Error connecting to database: {err}")


class VerifyToken(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(VerifyToken, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        token = auth_header.split(" ")[1] if " " in auth_header else None
        if not token:
            raise HTTPException(status_code=401, detail="Token missing")

        try:
            
            secret_key = os.getenv("ACCESS_TOKEN_SECRET")
            if not secret_key:
                raise HTTPException(status_code=500, detail="Server misconfigured: ACCESS_TOKEN_SECRET missing")

           
            decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
            
            
            request.state.email = decoded.get("email")
            if not request.state.email:
                raise HTTPException(status_code=403, detail="Email missing in token payload")

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=403, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=403, detail="Invalid token")


# Use middleware in endpoints
security = VerifyToken()

# Save prediction result to database
def save_prediction_to_db(email: str, prediction_data: dict):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        query = """
            INSERT INTO predictions (email, nama_penyakit, severity, severityLevel, suggestion, createdAt)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            email,
            prediction_data["nama_penyakit"],
            prediction_data["severity"],
            prediction_data["severityLevel"],
            ", ".join(prediction_data["suggestion"]),
            prediction_data["createdAt"]
        ))
        connection.commit()
    except mysql.connector.Error as err:
        connection.rollback()
        raise Exception(f"Failed to save prediction: {err}")
    finally:
        cursor.close()
        connection.close()

# Utility to preprocess image
def preprocess_image(image):
    tensor = tf.image.resize(image, (224, 224))
    tensor = tf.expand_dims(tensor, axis=0) 
    tensor = tf.cast(tensor, tf.float32) / 255.0  
    return tensor

# Prediction functions
def predict_severity(interpreter, image):
    tensor = preprocess_image(image)
    interpreter.set_tensor(input_details_severity[0]['index'], tensor.numpy())
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details_severity[0]['index'])[0]
    confidence_score = float(np.max(predictions)) * 100
    classes = ["sedang", "ringan", "Parah"]
    severity_level = int(np.argmax(predictions))
    label = classes[severity_level]
    return {
        "confidenceScore": confidence_score,
        "severityLevel": severity_level,
        "label": label,
    }

def predict_disease(interpreter, image):
    tensor = preprocess_image(image)
    interpreter.set_tensor(input_details_disease[0]['index'], tensor.numpy())
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details_disease[0]['index'])[0]
    confidence_score = float(np.max(predictions)) * 100
    disease_classes = [
        "Acne and Rosacea", "Atopic Dermatitis", "Herpes",
        "Psoriasis", "Tinea Ringworm", "Wart"
    ]
    disease_label = disease_classes[int(np.argmax(predictions))]
    return {
        "confidenceScore": confidence_score,
        "label": disease_label,
        "suggestion": ["No specific suggestions available."],  # Placeholder
    }

@app.post("/predict")
async def post_predict_handler(
    request: Request,
    file: UploadFile = File(...),
    token: str = Depends(security)
):

    try:
        # Get email from request.state set by VerifyToken middleware
        email = request.state.email

        # Read image from file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")

        # Convert image to numpy array
        image_array = tf.convert_to_tensor(np.array(image))

        # Perform predictions
        severity_result = predict_severity(interpreter_severity, image_array)
        disease_result = predict_disease(interpreter_disease, image_array)

        # Combine results
        response_data = {
            "id": str(uuid.uuid4()),
            "nama_penyakit": disease_result["label"],
            "severity": severity_result["label"],
            "severityLevel": severity_result["severityLevel"],
            "suggestion": disease_result["suggestion"],
            "confidenceScore": {
                "severity": severity_result["confidenceScore"],
                "disease": disease_result["confidenceScore"],
            },
            "createdAt": datetime.utcnow().isoformat(),
        }

        # Save prediction to database
        save_prediction_to_db(email, response_data)

        return JSONResponse(
            content={
                "status": "success",
                "message": "Prediction completed successfully.",
                "data": response_data,
            },
            status_code=201,
        )
    except HTTPException as e:
        return JSONResponse(
            content={"status": "error", "message": e.detail},
            status_code=e.status_code,
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": f"Prediction failed: {str(e)}"},
            status_code=500,
        )

@app.get("/")
def home():
    return {"message": "API is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
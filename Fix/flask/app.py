import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from datetime import datetime
from pathlib import Path
from utils import training, predict, FuzzyLogic

from controller.captureImageController import capture_image

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder="public", static_url_path="/public")

# Custom environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))

app.static_folder = "public"
app.static_url_path = "/public"


@app.route("/")
def home():
    return f"Welcome to the Flask App!"


@app.route("/capture-image")
def capture_image_route():
    filepath = capture_image()
    if type(filepath) is str:
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Success capturing image",
                    "filepath": filepath,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_url": f"/public/images/{os.path.basename(filepath)}",
                }
            ),
            200,
        )
    else:
        return filepath


# Route for get latest image
@app.route("/latest-image")
def latest_image():
    images_dir = Path("public/images")
    if not images_dir.exists():
        return (
            jsonify({"status": "error", "message": "Images directory does not exist"}),
            404,
        )

    latest_image = max(images_dir.glob("*.jpeg"), key=os.path.getctime, default=None)
    if latest_image is None:
        return (
            jsonify({"status": "error", "message": "No images found"}),
            404,
        )

    return (
        jsonify(
            {
                "status": "success",
                "image_url": f"/public/images/{latest_image.name}",
                "timestamp": latest_image.stat().st_ctime,
            }
        ),
        200,
    )


@app.route("/delay-spray", methods=["POST"])
def delay_spray_route():
    air_temperature = request.json.get("temperature")
    humidity = request.json.get("humidity")
    if air_temperature is None or humidity is None:
        return (
            jsonify(
                {"status": "error", "message": "Missing airTemperature or humidity"}
            ),
            400,
        )

    delay = FuzzyLogic.CalculateSprayingDelay(air_temperature, humidity)

    if delay is None:
        return jsonify({"status": "error", "message": "Failed to calculate delay"}), 500

    return jsonify({"status": "success", "delay": delay}), 200


@app.route("/train", methods=["POST"])
def train_model():
    is_hyperparameter_tuning = request.json.get("hyperparameter_tuning", False)
    if not isinstance(is_hyperparameter_tuning, bool):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "hyperparameter_tuning must be a boolean",
                }
            ),
            400,
        )

    # Temporary
    # Always is_hyperparameter_tuning to False
    if is_hyperparameter_tuning:
        is_hyperparameter_tuning = False

    try:
        start_training = datetime.now()

        training.training_model(is_hyperparameter_tuning=is_hyperparameter_tuning)

        training_duration = datetime.now() - start_training
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Training failed: {str(e)}",
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "status": "success",
                "message": "Training completed",
                "duration": str(training_duration),
            }
        ),
        200,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    start_prediction = datetime.now()
    start_date = request.json.get("start_date")

    if not start_date:
        return (
            jsonify({"status": "error", "message": "Missing start_date parameter"}),
            400,
        )

    try:
        prediction = predict.predict(start_date)

        if prediction is None:
            return (
                jsonify({"status": "error", "message": "Failed to get prediction"}),
                500,
            )

        # Save the prediction to a file
        filepath = Path("public/prediction/data.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            prediction.to_csv(filepath, mode="a", header=False)
        else:
            prediction.to_csv(filepath)

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Prediction failed: {str(e)}",
                }
            ),
            500,
        )

    prediction_duration = datetime.now() - start_prediction

    return (
        jsonify(
            {
                "status": "success",
                "message": "Prediction completed",
                "duration": str(prediction_duration),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=app.config["DEBUG"])

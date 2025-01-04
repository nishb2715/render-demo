from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = "migraine_trigger_model.pkl"
try:
    model = pickle.load(open(model_path, "rb"))
   
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Extract form inputs
        patient_id = request.form.get("patient_id")
        diet = request.form.get("diet")
        visual = request.form.get("visual")
        sleep_duration = request.form.get("sleep_duration")

        # Validate inputs
        if not diet or not visual or not sleep_duration:
            return jsonify({"error": "All fields are required except 'gender'."}), 400

        # Map inputs to numerical values
        diet_map = {"yes": 1, "no": 0}
        visual_map = {"yes": 1, "no": 0}

        diet = diet_map.get(diet.lower())
        visual = visual_map.get(visual.lower())

        # Validate sleep duration
        try:
            sleep_duration = float(sleep_duration)
            if not (0 <= sleep_duration <= 14):
                return jsonify({"error": "Sleep duration must be between 0 and 14 hours."}), 400
        except ValueError:
            return jsonify({"error": "Invalid sleep duration value."}), 400

        # Prepare input data for prediction
        input_features = np.array([[diet, visual, sleep_duration]])

        # Make prediction
        prediction = model.predict(input_features)

        # Return prediction
        result = {
            "patient_id": patient_id,
            "prediction": "Migraine Trigger Detected" if prediction[0] == 1 else "No Migraine Trigger"
        }
        return jsonify(result)

    except Exception as e:
        # Handle any other errors
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

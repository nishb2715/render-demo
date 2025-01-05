from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)


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
        
        patient_id = request.form.get("patient_id")
        diet = request.form.get("diet")
        visual = request.form.get("visual")
        sleep_duration = request.form.get("sleep_duration")

        
        if not diet or not visual or not sleep_duration:
            return jsonify({"error": "All fields are required except 'gender'."}), 400

        
        diet_map = {"yes": 1, "no": 0}
        visual_map = {"yes": 1, "no": 0}

        diet = diet_map.get(diet.lower())
        visual = visual_map.get(visual.lower())

        
        try:
            sleep_duration = float(sleep_duration)
            if not (0 <= sleep_duration <= 14):
                return jsonify({"error": "Sleep duration must be between 0 and 14 hours."}), 400
        except ValueError:
            return jsonify({"error": "Invalid sleep duration value."}), 400

        
        input_features = np.array([[diet, visual, sleep_duration]])

        
        prediction = model.predict(input_features)

        
        result = {
            "patient_id": patient_id,
            "prediction": "Migraine Trigger Detected" if prediction[0] == 1 else "No Migraine Trigger"
        }
        return jsonify(result)

    except Exception as e:
     
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

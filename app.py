from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Trained Model and Scaler
MODEL_PATH = "model/floods.save"
SCALER_PATH = "model/scaler.save"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None
    print("Warning: Model or Scaler not found! Please run train_model.py first.")

FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_input")
def predict_input():
    return render_template("predict_input.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return "Model or Scaler not loaded. Please train the model first.", 500
    
    try:
        # Collect values in order
        values = []
        for feature in FEATURES:
            val = request.form.get(feature)
            if val is None or val == "":
                return f"Missing value for feature: {feature}", 400
            values.append(float(val))
        
        # Scale values and predict
        input_array = np.array(values).reshape(1, -1)
        scaled_values = scaler.transform(input_array)
        
        # Binary prediction
        prediction = model.predict(scaled_values)[0]
        
        # Probability calculation (Premium feature)
        probability = model.predict_proba(scaled_values)[0][1]
        prob_percent = round(probability * 100, 2)
        
        # Render specific pages depending on binary outcome
        if prediction == 1:
            return render_template(
                "flood_chance.html",
                prob=prob_percent,
                inputs=dict(zip(FEATURES, values))
            )
        else:
            return render_template(
                "no_flood_chance.html",
                prob=prob_percent,
                inputs=dict(zip(FEATURES, values))
            )
            
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500

@app.route("/validation")
def validation():
    # Hardcoded model metrics based on seed=27 evaluation
    metrics = {
        "Decision Tree": {"accuracy": 69.79, "status": "Baseline"},
        "K-Nearest Neighbors": {"accuracy": 84.65, "status": "Moderate"},
        "Random Forest": {"accuracy": 89.85, "status": "High"},
        "XGBoost (Best)": {"accuracy": 96.55, "status": "Outstanding"}
    }
    return render_template("validation.html", metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
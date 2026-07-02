from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model/floods.save")
scaler = joblib.load("model/scaler.save")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = []

    values.append(float(request.form["MonsoonIntensity"]))
    values.append(float(request.form["TopographyDrainage"]))
    values.append(float(request.form["RiverManagement"]))
    values.append(float(request.form["Deforestation"]))
    values.append(float(request.form["Urbanization"]))
    values.append(float(request.form["ClimateChange"]))
    values.append(float(request.form["DamsQuality"]))
    values.append(float(request.form["Siltation"]))
    values.append(float(request.form["AgriculturalPractices"]))
    values.append(float(request.form["Encroachments"]))
    values.append(float(request.form["IneffectiveDisasterPreparedness"]))
    values.append(float(request.form["DrainageSystems"]))
    values.append(float(request.form["CoastalVulnerability"]))
    values.append(float(request.form["Landslides"]))
    values.append(float(request.form["Watersheds"]))
    values.append(float(request.form["DeterioratingInfrastructure"]))
    values.append(float(request.form["PopulationScore"]))
    values.append(float(request.form["WetlandLoss"]))
    values.append(float(request.form["InadequatePlanning"]))
    values.append(float(request.form["PoliticalFactors"]))

    scaled_values = scaler.transform(np.array(values).reshape(1, -1))
    prediction = model.predict(scaled_values)

    if prediction[0] == 1:
        result = "⚠️ High Flood Risk"
        color = "high"
    else:
        result = "✅ Low Flood Risk"
        color = "low"
    return render_template("result.html", result=result, color=color)


if __name__ == "__main__":
    app.run(debug=True)
import os
import joblib
import numpy as np

# Feature list used by the model
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]

MODEL_PATH = "model/floods.save"
SCALER_PATH = "model/scaler.save"

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found! Please run train_model.py first.")
        return None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def run_prediction(model, scaler, feature_values):
    # Scale values and predict
    input_array = np.array(feature_values).reshape(1, -1)
    scaled_values = scaler.transform(input_array)
    
    # Binary prediction & probability
    prediction = model.predict(scaled_values)[0]
    probability = model.predict_proba(scaled_values)[0][1]
    
    return prediction, probability

def main():
    model, scaler = load_model()
    if not model or not scaler:
        return

    print("--- Flood Prediction CLI Tool ---")
    print("Choose an option:")
    print("1. Run with average/sample values (All features set to 5)")
    print("2. Enter feature values interactively")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Average/sample values (5 for each feature)
        sample_values = [5] * len(FEATURES)
        prediction, probability = run_prediction(model, scaler, sample_values)
        
        print("\n--- Results (Sample Input: All features set to 5) ---")
        for feat, val in zip(FEATURES, sample_values):
            print(f"  {feat}: {val}")
        print("-" * 50)
        print(f"Flood Risk Prediction: {'HIGH RISK (Flood Chance)' if prediction == 1 else 'SAFE (No Flood Chance)'}")
        print(f"Flood Probability: {probability * 100:.2f}%")
        
    elif choice == "2":
        print("\nPlease enter a score between 0 and 10 for each of the following factors:")
        user_values = []
        for feature in FEATURES:
            while True:
                try:
                    val = input(f"  {feature} score: ").strip()
                    val_float = float(val)
                    if 0 <= val_float <= 20: # allow slightly larger values as model was trained on max values up to 18
                        user_values.append(val_float)
                        break
                    else:
                        print("    Please enter a number, typically between 0 and 10.")
                except ValueError:
                    print("    Invalid input. Please enter a valid number.")
        
        prediction, probability = run_prediction(model, scaler, user_values)
        
        print("\n--- Prediction Results ---")
        print(f"Flood Risk Prediction: {'HIGH RISK (Flood Chance)' if prediction == 1 else 'SAFE (No Flood Chance)'}")
        print(f"Flood Probability: {probability * 100:.2f}%")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()

"""
Student Placement Prediction - Flask Web App
----------------------------------------------
This file:
1. Loads the trained RandomForest model (Student_placement_model.pkl)
2. Rebuilds the SAME StandardScaler used during training (fit on the
   same 80% training split, with the same random_state=42) so that
   incoming form data is scaled exactly like it was during training.
3. Defines two routes:
   - "/"        -> shows the input form
   - "/predict" -> receives form data, scales it, predicts, shows result
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---------- 1. Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Student_placement_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "model", "placementdata.csv")

# ---------- 2. Load the trained model ----------
model = joblib.load(MODEL_PATH)

# The exact feature order the model was trained on (stored inside the .pkl)
FEATURE_ORDER = list(model.feature_names_in_)

# ---------- 3. Rebuild the training-time scaler ----------
# NOTE: the notebook didn't save the scaler, so we recreate it here using
# the exact same steps (same columns dropped, same split, same random_state)
# so the numbers match what the model expects.
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["StudentID"])
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
df["PlacementTraining"] = df["PlacementTraining"].map({"Yes": 1, "No": 0})
df["PlacementStatus"] = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)  # fit only on training data, same as the notebook


# ---------- 4. Routes ----------
@app.route("/")
def home():
    """Show the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Read form values, scale them, run the model, show the result."""

    # --- Read raw form inputs (all come in as strings) ---
    form = request.form

    input_dict = {
        "CGPA": float(form.get("cgpa")),
        "Internships": int(form.get("internships")),
        "Projects": int(form.get("projects")),
        "Workshops/Certifications": int(form.get("workshops")),
        "AptitudeTestScore": float(form.get("aptitude")),
        "SoftSkillsRating": float(form.get("softskills")),
        "ExtracurricularActivities": 1 if form.get("extracurricular") == "Yes" else 0,
        "PlacementTraining": 1 if form.get("training") == "Yes" else 0,
        "SSC_Marks": float(form.get("ssc")),
        "HSC_Marks": float(form.get("hsc")),
    }

    # --- Build a DataFrame in the EXACT column order the model expects ---
    input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

    # --- Scale using the same scaler statistics as training ---
    # (keep it as a DataFrame with column names to avoid sklearn warnings)
    input_scaled = pd.DataFrame(
        scaler.transform(input_df), columns=FEATURE_ORDER
    )

    # --- Predict ---
    prediction = model.predict(input_scaled)[0]  # 0 or 1
    probability = model.predict_proba(input_scaled)[0][1]  # probability of "Placed"

    result = "Placed" if prediction == 1 else "Not Placed"
    confidence = round(probability * 100, 2) if prediction == 1 else round((1 - probability) * 100, 2)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        inputs=input_dict,
    )


if __name__ == "__main__":
    # debug=True auto-reloads the server when you save changes (dev only!)
    app.run(debug=True)

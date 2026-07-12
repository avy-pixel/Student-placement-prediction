from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("Student_placement_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    # Receive data from form
    cgpa = float(request.form.get("cgpa"))
    internships = int(request.form.get("internships"))
    projects = int(request.form.get("projects"))
    workshops = int(request.form.get("workshops"))
    aptitude = float(request.form.get("aptitude"))
    softskills = float(request.form.get("softskills"))
    extracurricular = int(request.form.get("extracurricular"))
    training = int(request.form.get("training"))
    ssc = float(request.form.get("ssc"))
    hsc = float(request.form.get("hsc"))

    # -----------------------------
    # Server-side Validation
    # -----------------------------
    if not (0 <= cgpa <= 10):
        return "CGPA must be between 0 and 10."

    if internships < 0:
        return "Internships cannot be negative."

    if projects < 0:
        return "Projects cannot be negative."

    if workshops < 0:
        return "Workshops cannot be negative."

    if not (0 <= aptitude <= 100):
        return "Aptitude score must be between 0 and 100."

    if not (1 <= softskills <= 10):
        return "Soft Skills rating must be between 1 and 10."

    if not (0 <= ssc <= 100):
        return "10th percentage must be between 0 and 100."

    if not (0 <= hsc <= 100):
        return "12th percentage must be between 0 and 100."

    # -----------------------------
    # Create DataFrame
    # -----------------------------
    features = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Workshops/Certifications": [workshops],
        "AptitudeTestScore": [aptitude],
        "SoftSkillsRating": [softskills],
        "ExtracurricularActivities": [extracurricular],
        "PlacementTraining": [training],
        "SSC_Marks": [ssc],
        "HSC_Marks": [hsc]
    })

    # Scale data
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)[0]

    # Prediction confidence
    probabilities = model.predict_proba(scaled_features)[0]

    if prediction == 1:
        probability = probabilities[1] * 100
    else:
        probability = probabilities[0] * 100

    # -----------------------------
    # Suggestions
    # -----------------------------
    suggestions = []

    if cgpa < 7.5:
        suggestions.append(
            "Improve your CGPA. A CGPA above 7.5 can strengthen your placement profile."
        )

    if internships == 0:
        suggestions.append(
            "Complete at least one internship to gain practical experience."
        )

    if projects < 2:
        suggestions.append(
            "Build more academic or personal projects to showcase your skills."
        )

    if workshops < 2:
        suggestions.append(
            "Attend workshops or earn certifications to improve your technical knowledge."
        )

    if aptitude < 70:
        suggestions.append(
            "Practice aptitude questions regularly to improve your test performance."
        )

    if softskills < 7:
        suggestions.append(
            "Work on communication, teamwork, and interview skills."
        )

    if extracurricular == 0:
        suggestions.append(
            "Participate in extracurricular activities to demonstrate leadership and teamwork."
        )

    if training == 0:
        suggestions.append(
            "Join a placement training program to prepare for interviews."
        )

    if ssc < 70:
        suggestions.append(
            "Many companies consider academic consistency, so strong fundamentals are beneficial."
        )

    if hsc < 70:
        suggestions.append(
            "Improving subject knowledge can help with technical interviews."
        )

    # Result
    if prediction == 1:
        result = "Likely to be Placed"
    else:
        result = "Not Likely to be Placed"

    return render_template(
        "result.html",
        result=result,
        prediction=prediction,
        probability=round(probability, 2),
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)
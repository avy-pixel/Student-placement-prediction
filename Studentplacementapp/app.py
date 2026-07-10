from flask import Flask,render_template,request
import joblib
app = Flask(__name__)
#Load model and scaler only once when Flask starts
model=joblib.load("Student_placement_model.pkl")
scaler=joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    # Receive data from the form
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

    # Arrange features in the same order used during training
    features = [[
        cgpa,
        internships,
        projects,
        workshops,
        aptitude,
        softskills,
        extracurricular,
        training,
        ssc,
        hsc
    ]]

    # Scale the data
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)[0]
    probability=model.predict_proba(scaled_features)[0][1]*100

    suggestions=[]
    if cgpa < 7.5:
       suggestions.append("Improve your CGPA. A CGPA above 7.5 can strengthen your placement profile.")
    if internships == 0:
        suggestions.append("Complete at least one internship to gain practical experience.")
    if projects < 2:
        suggestions.append("Build more academic or personal projects to showcase your skills.")

    if workshops < 2:
        suggestions.append("Attend workshops or earn certifications to improve your technical knowledge.")

    if aptitude < 70:
        suggestions.append("Practice aptitude questions regularly to improve your test performance.")

    if softskills < 7:
        suggestions.append("Work on communication, teamwork, and interview skills.")

    if extracurricular == 0:
        suggestions.append("Participate in extracurricular activities to demonstrate leadership and teamwork.")

    if training == 0:
        suggestions.append("Join a placement training program to prepare for interviews.")

    if ssc < 70:
        suggestions.append("Many companies consider academic consistency, so strong fundamentals are beneficial.")

    if hsc < 70:
        suggestions.append("Improving subject knowledge can help with technical interviews.")
    
    result="Likely to be Placed" if prediction == 1 else "Not Likely to be Placed"



    return render_template("result.html", result=result, suggestions=suggestions ,probability=round(probability,2))
if __name__=="__main__":
    app.run(debug=True)
    

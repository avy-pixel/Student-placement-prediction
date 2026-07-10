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

    return f"""
    CGPA: {cgpa}<br>
    Internships: {internships}<br>
    Projects: {projects}<br>
    Workshops: {workshops}<br>
    Aptitude: {aptitude}<br>
    Soft Skills: {softskills}<br>
    Extracurricular: {extracurricular}<br>
    Placement Training: {training}<br>
    SSC: {ssc}<br>
    HSC: {hsc}
    """
     

if __name__=="__main__":
    app.run(debug=True)
    

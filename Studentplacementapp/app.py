from flask import Flask
import joblib
app = Flask(__name__)
#Load model and scaler only once when Flask starts
model=joblib.load("Student_placement_model.pkl")
scaler=joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "<h1>Student Placement Prediction</h1>"

if __name__=="__main__":
    app.run(debug=True)
    

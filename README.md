# Student Placement Prediction

Predicts whether a student is likely to be placed based on academics, experience, and test performance — trained on 10,000 student records and served through a Flask web app.

Enter a student's CGPA, internships, projects, test scores, and more, and the app returns a placement verdict, a confidence score, and a list of areas to improve.

## How it works

1. `Studentdata/Notebook.ipynb` walks through the full data science pipeline: cleaning `placementdata.csv`, feature scaling with `StandardScaler`, and training/comparing three classifiers — Logistic Regression, Decision Tree, and Random Forest — using accuracy, precision, recall, F1, and ROC-AUC.
2. The best-performing model (Random Forest) and its fitted scaler are exported as `Student_placement_model.pkl` and `scaler.pkl`.
3. `Studentplacementapp/app.py` is a Flask app that loads both artifacts, validates form input, scales it, and returns a prediction with probability and personalized suggestions.

## Features used

| Feature | Description |
|---|---|
| CGPA | Cumulative GPA (0–10) |
| Internships | Number of internships completed |
| Projects | Number of academic/personal projects completed |
| Workshops/Certifications | Number attended/earned |
| AptitudeTestScore | Aptitude test score (0–100) |
| SoftSkillsRating | Self/rated soft skills score (1–10) |
| ExtracurricularActivities | Participated (Yes/No) |
| PlacementTraining | Completed placement training (Yes/No) |
| SSC_Marks | 10th grade percentage |
| HSC_Marks | 12th grade percentage |

## Project structure
Student-placement-prediction/
├── Studentdata/
│   ├── Notebook.ipynb        # EDA, model training & evaluation
│   └── placementdata.csv     # Dataset (10,000 records)
├── Studentplacementapp/
│   ├── app.py                 # Flask app
│   ├── Student_placement_model.pkl
│   ├── scaler.pkl
│   ├── static/style.css
│   └── templates/
│       ├── index.html
│       ├── predict.html
│       └── result.html
└── requirements.txt
## Running locally

```bash
git clone https://github.com/avy-pixel/Student-placement-prediction.git
cd Student-placement-prediction
pip install -r requirements.txt
cd Studentplacementapp
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> The UI pulls its fonts (Fraunces, IBM Plex Sans/Mono) from Google Fonts, so an internet connection is needed for the styling to render as intended — it falls back to system fonts offline.

## Tech stack

- **Modeling:** scikit-learn (Random Forest Classifier), pandas, numpy
- **Backend:** Flask
- **Frontend:** HTML/CSS (Jinja2 templates, no JS framework)
- **Deployment:** gunicorn-ready

## Notes

- Server-side validation enforces sane ranges for all numeric inputs (e.g. CGPA between 0–10, percentages between 0–100).
- The result page's suggestions are rule-based (not model-derived) and highlight the weakest areas of a given student's profile.

## License

No license file is currently included — add one if you intend for others to reuse this code.
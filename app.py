
from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("normalizer.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rc_acc_68.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template_string(open("templates/index.html").read())

@app.route("/portfolio")
def portfolio():
    return render_template_string(open("templates/portfolio.html").read())

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            vals = [float(request.form[k]) for k in ["ALT", "AST", "ALP", "Bilirubin", "Albumin", "Age"]]
            norm_vals = scaler.transform([vals])
            pred = model.predict(norm_vals)[0]
            result = "⚠️ Likely Liver Disease" if pred == 1 else "✅ Healthy Liver"
        except:
            result = "Invalid input"
    return render_template_string(open("templates/form.html").read(), result=result)

if __name__ == "__main__":
    app.run(debug=True)

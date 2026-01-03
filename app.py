from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ]

    prediction = model.predict([features])[0]

    classes = ["Setosa", "Versicolor", "Virginica"]
    flower_name = classes[prediction]

    image_map = {
        "Setosa": "images/setosa.jpg",
        "Versicolor": "images/versicolor.jpg",
        "Virginica": "images/virginica.jpg"
    }

    return render_template(
        "index.html",
        prediction_text=f"Iris Flower Type: {flower_name}",
        flower_image=image_map[flower_name]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

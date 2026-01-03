from flask import Flask, render_template, request
import joblib

app = Flask(__name__, static_folder="static", static_url_path="/static")


model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    # Simple validation (Iris flower range check)
    if not (4 <= sl <= 8 and 2 <= sw <= 5 and 1 <= pl <= 7 and 0.1 <= pw <= 3):
        return render_template(
            "index.html",
            prediction_text="❌ This is NOT an Iris flower",
            flower_image=None
        )

    prediction = model.predict([[sl, sw, pl, pw]])[0]

    flower_map = {
        0: ("Iris Setosa", "images/setosa.jpg"),
        1: ("Iris Versicolor", "images/versicolor.jpg"),
        2: ("Iris Virginica", "images/virginica.jpg")
    }

    flower_name, flower_image = flower_map[prediction]

    return render_template(
        "index.html",
        prediction_text=f"🌸 Predicted Flower: {flower_name}",
        flower_image=flower_image
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



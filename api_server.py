"""API server example"""

#
# Usage from command line:

#para powershell
# Invoke-WebRequest -Uri http://127.0.0.1:5000 -Method POST -Headers @{"Content-Type" = "application/json"} -Body '{"bathrooms": 2, "bedrooms": 3, "sqft_living": 1800, "sqft_lot": 2200, "floors": 1, "waterfront": 1, "condition": 3}'
#

import pickle

import pandas as pd
from flask import Flask, request

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"


FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])
def index():
    """API function"""

    args = request.json
    filt_args = {key: [int(args[key])] for key in FEATURES}
    df = pd.DataFrame.from_dict(filt_args)

    with open("house_predictor.pickle", "rb") as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(df)

    return str(prediction[0][0])


if __name__ == "__main__":
    app.run(debug=True)
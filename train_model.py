"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

#montamos el modelo de regresión lineal.
estimator = LinearRegression()

#ajustamos el modelo segun las caracteristicas para que nos de el precio.
estimator.fit(features, target)

#hacemos esto para verlo pero no se usa generalmente
print(estimator.coef_)
print(estimator.intercept_)
#es probable que se borre

#con este bloque podemos guardar el modelo para poder llevarlo a producción
with open("house_predictor.pickle", "wb") as file:
    pickle.dump(estimator, file)
    
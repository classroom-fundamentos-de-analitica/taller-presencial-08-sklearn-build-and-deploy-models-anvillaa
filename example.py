import pickle

#abre el modelo en binario lo lee y lo carga en la variable loaded_model
with open("house_predictor.pickle", "rb") as file:
    loaded_model = pickle.load(file)

print(loaded_model.coef_)
print(loaded_model.intercept_)

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

modelo = joblib.load("models/best_model.pkl")

print("Modelo cargado correctamente")

df = pd.read_csv("WineQT.csv")

X = df.drop(columns=["quality", "Id"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

predicciones = modelo.predict(X_test)

print("Predicciones realizadas correctamente")

accuracy = accuracy_score(y_test, predicciones)
precision = precision_score(y_test, predicciones, average="weighted", zero_division=0)
recall = recall_score(y_test, predicciones, average="weighted", zero_division=0)
f1 = f1_score(y_test, predicciones, average="weighted", zero_division=0)

print("\nMétricas finales del modelo:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

matriz = confusion_matrix(y_test, predicciones)

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=modelo.classes_
)

visualizacion.plot(cmap="Blues")
plt.title("Matriz de confusión - Clasificación de vinos")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

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

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(X_train, y_train)

modelo_lr = LogisticRegression(
    max_iter=5000
)

modelo_lr.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, pred_rf)
precision_rf = precision_score(y_test, pred_rf, average="weighted", zero_division=0)
recall_rf = recall_score(y_test, pred_rf, average="weighted", zero_division=0)
f1_rf = f1_score(y_test, pred_rf, average="weighted", zero_division=0)

print("\nResultados de Random Forest:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1:", f1_rf)

pred_lr = modelo_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, pred_lr)
precision_lr = precision_score(y_test, pred_lr, average="weighted", zero_division=0)
recall_lr = recall_score(y_test, pred_lr, average="weighted", zero_division=0)
f1_lr = f1_score(y_test, pred_lr, average="weighted", zero_division=0)

print("\nResultados de Regresión Logística:")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1:", f1_lr)

import os

os.makedirs("models", exist_ok=True)

if f1_rf >= f1_lr:
    mejor_modelo = modelo_rf
    nombre_modelo = "Random Forest"
else:
    mejor_modelo = modelo_lr
    nombre_modelo = "Regresión Logística"

joblib.dump(mejor_modelo, "models/best_model.pkl")

print("\nMejor modelo:", nombre_modelo)
print("Modelo guardado en models/best_model.pkl")

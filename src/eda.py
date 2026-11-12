import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WineQT.csv")

print("Estadísticas descriptivas:")
print(df.describe())

print("\nValores faltantes:")
print(df.isnull().sum())

print("\nGenerando histogramas...")

df.drop(columns=["Id"]).hist(figsize=(14, 10))
plt.tight_layout()
plt.show()

print("\nGenerando boxplots...")

df.drop(columns=["Id"]).boxplot(figsize=(14, 8), rot=90)
plt.tight_layout()
plt.show()

print("\nGenerando matriz de correlación...")

plt.figure(figsize=(12, 10))
sns.heatmap(
    df.drop(columns=["Id"]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Matriz de correlación de vinos")
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/supply_clean.csv")

plt.figure(figsize=(10,6))
sns.histplot(df["Population_2025"], bins=40)
plt.title("Population Distribution")
plt.tight_layout()
plt.savefig("data/population_distribution.png")

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Median_Age", y="Fert._Rate")
plt.title("Median Age vs Fertility Rate")
plt.tight_layout()
plt.savefig("data/age_vs_fertility.png")

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("data/correlation_matrix.png")

print("eda_complete")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("datasets/ml_preprocessed/all_sources.csv")
print(df.info())
print(df.describe())

corr = df.corr(numeric_only=True)
os.makedirs("datasets/ml_preprocessed/plots", exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.tight_layout()
plt.savefig("datasets/ml_preprocessed/plots/feature_correlation.png")
plt.close()

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(col)
    plt.tight_layout()
    plt.savefig(f"datasets/ml_preprocessed/plots/{col}_hist.png")
    plt.close()

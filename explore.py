import pandas as pd

df = pd.read_csv("Training.csv")

# Drop the known empty trailing column from this dataset, if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Shape:", df.shape)
print("\nColumns (first 10):", df.columns[:10].tolist())
print("\nTarget column unique values:", df['prognosis'].nunique())
print("\nAny missing values?", df.isnull().sum().sum())
print("\nClass distribution:\n", df['prognosis'].value_counts())
import matplotlib.pyplot as plt

# Class distribution plot
df['prognosis'].value_counts().plot(kind='bar', figsize=(18, 6))
plt.title("Disease Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
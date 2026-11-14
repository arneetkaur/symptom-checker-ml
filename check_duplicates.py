import pandas as pd

df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Total rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("Unique rows:", len(df.drop_duplicates()))

unique_per_class = df.drop_duplicates().groupby('prognosis').size()
print("\nUnique symptom patterns per disease:")
print(unique_per_class.describe())
print("\nFewest unique patterns:")
print(unique_per_class.sort_values().head(5))
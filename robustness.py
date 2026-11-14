import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, pairwise_distances

df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=['prognosis']).values
y = df['prognosis'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 1) How close is each test row to its nearest training row?
d = pairwise_distances(X_test, X_train, metric='hamming') * X.shape[1]
nearest = d.min(axis=1)
print("Symptoms differing from nearest training row:")
print("  mean:", nearest.mean().round(2), "| max:", nearest.max())

# 2) Accuracy when only a fraction of reported symptoms is kept
rng = np.random.default_rng(42)
def keep_fraction(arr, frac):
    mask = rng.random(arr.shape) < frac
    return arr * mask

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="linear"),
}

print("\nAccuracy when keeping only a fraction of symptoms:")
print(f"{'Model':<22}" + "".join(f"{f:>8}" for f in [1.0, 0.75, 0.5, 0.25]))
for name, m in models.items():
    m.fit(X_train, y_train)
    row = [accuracy_score(y_test, m.predict(keep_fraction(X_test, f)))
           for f in [1.0, 0.75, 0.5, 0.25]]
    print(f"{name:<22}" + "".join(f"{a:>8.3f}" for a in row))
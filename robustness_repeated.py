import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=['prognosis']).values
y = df['prognosis'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
}
fractions = [1.0, 0.75, 0.5, 0.25]
n_repeats = 20

def top_k_hits(proba, classes, y_true, k):
    top = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([y_true[i] in classes[top[i]] for i in range(len(y_true))])

rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    for f in fractions:
        top1, top3 = [], []
        for seed in range(n_repeats):
            rng = np.random.default_rng(seed)
            Xm = X_test * (rng.random(X_test.shape) < f)
            proba = m.predict_proba(Xm)
            top1.append(top_k_hits(proba, m.classes_, y_test, 1))
            top3.append(top_k_hits(proba, m.classes_, y_test, 3))
        rows.append({
            "model": name, "symptoms_kept": f,
            "top1_mean": np.mean(top1), "top1_std": np.std(top1),
            "top3_mean": np.mean(top3), "top3_std": np.std(top3),
        })

results = pd.DataFrame(rows).round(3)
print(results.to_string(index=False))
results.to_csv("robustness_results.csv", index=False)
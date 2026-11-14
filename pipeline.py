import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().reset_index(drop=True)

symptoms = [c for c in df.columns if c != 'prognosis']
X = df[symptoms].values
y = df['prognosis'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump({"model": model, "symptoms": symptoms}, "model.joblib")

def predict_top3(selected):
    x = np.zeros((1, len(symptoms)))
    for s in selected:
        x[0, symptoms.index(s)] = 1
    proba = model.predict_proba(x)[0]
    top = np.argsort(proba)[::-1][:3]
    results = []
    for i in top:
        contrib = model.coef_[i] * x[0]
        drivers = sorted(zip(symptoms, contrib), key=lambda t: -t[1])[:3]
        drivers = [(s, c) for s, c in drivers if c > 0]
        results.append((model.classes_[i], proba[i], drivers))
    return results

if __name__ == "__main__":
    demo = ["itching", "skin_rash", "nodal_skin_eruptions"]
    for disease, p, drivers in predict_top3(demo):
        print(f"{disease}: {p:.1%}")
        for s, c in drivers:
            print(f"    {s}: {c:+.2f}")
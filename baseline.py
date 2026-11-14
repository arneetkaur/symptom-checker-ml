import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load and clean
df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove duplicates BEFORE splitting
df = df.drop_duplicates().reset_index(drop=True)
print("Unique rows:", len(df))

X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Stratified split so every disease appears in the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Leakage check: this must print 0
train_rows = set(map(tuple, X_train.values))
overlap = sum(tuple(r) in train_rows for r in X_test.values)
print("Train/test overlapping rows:", overlap)
print("Train size:", len(X_train), "| Test size:", len(X_test))

# Baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nSingle split accuracy:", round(accuracy_score(y_test, pred), 3))
print("Single split macro-F1:", round(f1_score(y_test, pred, average='macro'), 3))

# 5-fold cross-validation for a more stable estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv)
print("\n5-fold CV accuracy:", scores.round(3))
print("Mean:", round(scores.mean(), 3), "| Std:", round(scores.std(), 3))
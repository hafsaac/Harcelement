# harcelement_ml2_cv.py
# - Split 80/20
# - Validation croisée 5-fold sur le train (80%)
# - Multiclasse sur "Categories"
# - Prédictions sur test -> predictions_base_test.csv

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_PATH = "Base_Entraînement - Feuille 1.csv"
TEST_PATH  = "Base_Test - Feuille 1.csv"

def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            f" Mets ce script dans le même dossier que les CSV, ou mets le chemin complet."
        )
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    return df

train_df = read_csv_safely(TRAIN_PATH)
test_df  = read_csv_safely(TEST_PATH)

train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

print("Colonnes train :", list(train_df.columns))
print("Colonnes test  :", list(test_df.columns))

TEXT_COL  = "Texte"       # ou "Traduction"
LABEL_COL = "Categories"  # multiclasse

for col in [TEXT_COL, LABEL_COL]:
    if col not in train_df.columns:
        raise ValueError(f"Colonne '{col}' introuvable dans TRAIN. Colonnes = {list(train_df.columns)}")

if TEXT_COL not in test_df.columns:
    raise ValueError(f"Colonne texte '{TEXT_COL}' introuvable dans TEST. Colonnes = {list(test_df.columns)}")

print(f"\n Colonne TEXTE utilisée : {TEXT_COL}")
print(f" Colonne LABEL utilisée : {LABEL_COL}\n")

train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
train_df[TEXT_COL]  = train_df[TEXT_COL].astype(str)
train_df[LABEL_COL] = train_df[LABEL_COL].astype(str)

X = train_df[TEXT_COL]
y = train_df[LABEL_COL]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,      # <-- 80/20 ICI
    random_state=42,
    stratify=y
)

print(f"📌 Taille train: {len(X_train)} | validation (20%): {len(X_valid)}")
print("📌 Classes:", sorted(y.unique()))

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=20000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        multi_class="auto"   # <-- MULTICLASSE ICI
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n🔁 Validation croisée (5-fold) sur le train (80%)...")
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
print("F1-macro par fold:", np.round(cv_scores, 4))
print("F1-macro moyen   :", round(cv_scores.mean(), 4))
print("Écart-type       :", round(cv_scores.std(), 4))

print("\n-> Entraînement du modèle sur le train (80%)...")
model.fit(X_train, y_train)
print(" Entraînement terminé.")

y_pred = model.predict(X_valid)

print("\n--- RAPPORT DE CLASSIFICATION (validation 20%) ---")
print(classification_report(y_valid, y_pred, digits=4, zero_division=0))

labels_order = sorted(y.unique())
cm = confusion_matrix(y_valid, y_pred, labels=labels_order)

print("\n--- MATRICE DE CONFUSION (validation 20%) ---")
print("Ordre des classes:", labels_order)
print(cm)

print("\n-> Entraînement final sur 100% du train...")
model.fit(X, y)

test_df = test_df.copy()
test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).fillna("")
test_pred = model.predict(test_df[TEXT_COL])

out = test_df.copy()
out["prediction"] = test_pred

OUT_PATH = "predictions_base_test.csv"
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n-> Fichier généré : {OUT_PATH}")
print("La colonne 'prediction' contient la classe prédite (multiclasse).")

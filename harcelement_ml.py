# ======================================================
# PROJET MACHINE LEARNING – DÉTECTION DU HARCELEMENT
# Base d'entraînement  →  Base de test
# ======================================================

# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# ==============================
# 2. CHARGEMENT DES DONNÉES
# ==============================

# IMPORTANT :
# Les fichiers doivent être dans le MÊME dossier que ce script
train_df = pd.read_excel("Base_Entrainement.xlsx")
test_df  = pd.read_excel("Base_Test.xlsx")

# Vérification des colonnes
print("Colonnes train :", train_df.columns.tolist())
print("Colonnes test  :", test_df.columns.tolist())


# ==============================
# 3. PRÉPARATION DES DONNÉES
# ==============================

# Colonnes correctes selon TES fichiers :
# - Texte   → message
# - Labels  → catégorie de harcèlement
X_train = train_df["Texte"].astype(str)
y_train = train_df["Labels"].astype(str)

X_test = test_df["Texte"].astype(str)
y_test = test_df["Labels"].astype(str)


# ==============================
# 4. PIPELINE MACHINE LEARNING
# (Lien entre train et test)
# ==============================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",   # tes textes sont en anglais
        max_features=5000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])


# ==============================
# 5. ENTRAÎNEMENT DU MODÈLE
# ==============================

print("\nEntraînement du modèle en cours...")
model.fit(X_train, y_train)
print("Entraînement terminé.")


# ==============================
# 6. PRÉDICTIONS SUR LA BASE TEST
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 7. ÉVALUATION DU MODÈLE
# ==============================

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_test, y_pred))

print("--- MATRICE DE CONFUSION ---")
print(confusion_matrix(y_test, y_pred))


# ==============================
# 8. EXPORT DES RÉSULTATS
# ==============================

resultats = test_df.copy()
resultats["Prediction"] = y_pred

resultats.to_excel("Base_Test_Predictions.xlsx", index=False)

print("\nFichier généré avec succès : Base_Test_Predictions.xlsx")

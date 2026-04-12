import sys
import joblib
import os

# =========================================================
# CHEMINS DES MODÈLES
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_BIN  = os.path.join(MODELS_DIR, "modele_reglog_binaire.pkl")
MODEL_CAT  = os.path.join(MODELS_DIR, "modele_reglog_multiclasse.pkl")

# =========================================================
# CHARGEMENT
# =========================================================
try:
    model_bin = joblib.load(MODEL_BIN)
    model_cat = joblib.load(MODEL_CAT)
except Exception as e:
    print(f"ERREUR_CHARGEMENT:{e}")
    sys.exit(1)

# =========================================================
# PRÉDICTION
# =========================================================
texte = sys.argv[1] if len(sys.argv) > 1 else ""

if not texte.strip():
    print("ERREUR_TEXTE:vide")
    sys.exit(1)

# Tâche binaire
pred_bin   = model_bin.predict([texte])[0]
proba_bin  = model_bin.predict_proba([texte])[0]
confiance  = round(float(max(proba_bin)) * 100, 1)

# Tâche multi-classes (seulement si haineux)
if pred_bin == "Haineux" or pred_bin == 1:
    pred_cat  = model_cat.predict([texte])[0]
    proba_cat = model_cat.predict_proba([texte])[0]
    confiance_cat = round(float(max(proba_cat)) * 100, 1)
else:
    pred_cat      = "Non applicable"
    confiance_cat = 0.0

# =========================================================
# SORTIE — format simple lisible par PHP
# =========================================================
print(f"BINAIRE:{pred_bin}")
print(f"CONFIANCE_BIN:{confiance}")
print(f"CATEGORIE:{pred_cat}")
print(f"CONFIANCE_CAT:{confiance_cat}")
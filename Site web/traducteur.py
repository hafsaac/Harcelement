# =========================================================
# TRADUCTEUR.PY — Transforme un message haineux en message bienveillant
# =========================================================

import sys
import re

# =========================================================
# DICTIONNAIRE DE REMPLACEMENT PAR CATÉGORIE
# =========================================================

remplacements = {
    # Mots ciblant des groupes → termes neutres
    "disabled":     "person with a disability",
    "cripple":      "person with a disability",
    "retard":       "person with different abilities",
    "mong":         "person with different abilities",
    "gay":          "LGBTQ+ person",
    "faggot":       "person",
    "fag":          "person",
    "queer":        "person",
    "dyke":         "person",
    "nigger":       "person",
    "nigga":        "person",
    "negro":        "person",
    "coon":         "person",
    "blck":         "Black person",
    "wetback":      "immigrant",
    "immigrant":    "person who moved here",
    "immigrants":   "people who moved here",
    "refugee":      "person seeking safety",
    "muzzie":       "Muslim person",
    "jihadi":       "person",
    "terrorist":    "person",
    "muslims":      "Muslim people",
    "muslim":       "Muslim person",
    "camel":        "person",
    "women":        "people",
    "woman":        "person",
    "bitch":        "person",
    "slut":         "person",
    "whore":        "person",
    "female":       "person",

    # Verbes / expressions violentes → formulations neutres
    "kill":         "disagree with",
    "shoot":        "oppose",
    "die":          "leave",
    "hate":         "strongly disagree with",
    "disgusting":   "concerning",
    "repulsive":    "troubling",
    "worthless":    "different",
    "stupid":       "mistaken",
    "idiot":        "person with a different view",
    "scum":         "person",
    "dirt":         "person",
    "rapist":       "person",
    "rapists":      "people",
    "executed":     "opposed",
    "contempt":     "disagreement",
    "despicable":   "concerning",
    "absolute":     "real",
    "absolutely":   "truly",
    "typical":      "common",
    "never":        "rarely",
    "nothing":      "little",
    "must":         "could",
    "deserve":      "might benefit from",
    "suffer":       "face challenges",
    "rid":          "move on from",
    "bunch":        "group",
    "worst":        "most challenging",
}

# =========================================================
# TRANSFORMATIONS STRUCTURELLES
# =========================================================

def adoucir_structure(texte):
    """Transforme les formulations agressives en formulations constructives"""

    # Supprimer les points d'exclamation multiples
    texte = re.sub(r'!+', '.', texte)

    # Supprimer les majuscules excessives (mots entiers en majuscules)
    mots = texte.split()
    mots_adoucis = []
    for mot in mots:
        if mot.isupper() and len(mot) > 2:
            mots_adoucis.append(mot.capitalize())
        else:
            mots_adoucis.append(mot)
    texte = ' '.join(mots_adoucis)

    # Remplacer "should not" par "could consider"
    texte = re.sub(r'should not', 'could reconsider whether', texte, flags=re.IGNORECASE)
    texte = re.sub(r'should never', 'rarely should', texte, flags=re.IGNORECASE)

    # Remplacer "are not" par "may not always be"
    texte = re.sub(r'are not', 'are not always', texte, flags=re.IGNORECASE)

    # Supprimer les insultes directes isolées
    texte = re.sub(r'\b(fuck|shit|damn|ass|bastard|crap)\b', '', texte, flags=re.IGNORECASE)

    # Nettoyer les espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()

    return texte

# =========================================================
# FONCTION PRINCIPALE
# =========================================================

def traduire(texte_original):
    texte = texte_original.lower()

    # Appliquer les remplacements mot par mot
    for mot_haineux, mot_bienveillant in remplacements.items():
        pattern = r'\b' + re.escape(mot_haineux) + r'\b'
        texte = re.sub(pattern, mot_bienveillant, texte, flags=re.IGNORECASE)

    # Adoucir la structure
    texte = adoucir_structure(texte)

    # Remettre la première lettre en majuscule
    if texte:
        texte = texte[0].upper() + texte[1:]

    # Ajouter un message de contexte
    intro = "💬 Version bienveillante : "

    return intro + texte

# =========================================================
# EXÉCUTION
# =========================================================

if __name__ == "__main__":
    texte = sys.argv[1] if len(sys.argv) > 1 else ""
    if not texte.strip():
        print("ERREUR:texte vide")
        sys.exit(1)
    print(traduire(texte))
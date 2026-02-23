import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import os

# --- ÉTAPE 1 : CHARGEMENT ---
try:
    df = pd.read_csv("train.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    
    # Visu 1 : Répartition
    plt.figure(figsize=(10, 6))
    # On remplace les vides par 'Non Haineux' pour le graphique
    df['Labels'] = df['Labels'].fillna('Non Haineux')
    sns.countplot(data=df, y='Labels', palette='viridis', hue='Labels', legend=False)
    plt.title("Répartition des messages par catégorie")
    plt.tight_layout()
    plt.savefig("graphique_repartition.png")
    print("✅ 1. Graphique répartition : OK")

    # Visu 2 : Nuage de mots
    texte_global = " ".join(df['Texte'].astype(str))
    nuage = WordCloud(width=800, height=400, background_color='white').generate(texte_global)
    plt.figure(figsize=(10, 5))
    plt.imshow(nuage)
    plt.axis("off")
    plt.savefig("nuage_mots.png")
    print("✅ 2. Nuage de mots : OK")

except Exception as e:
    print(f"❌ Erreur Train : {e}")

# --- ÉTAPE 3 : MATRICE DE CONFUSION (LA CORRECTION EST ICI) ---
try:
    if os.path.exists("preds.csv"):
        df_p = pd.read_csv("preds.csv")
        df_p.columns = df_p.columns.str.strip()
        
        # CORRECTION : On remplace les cases vides (NaN) par 'Non Haineux'
        # Et on force tout en texte (str) pour éviter l'erreur de tri
        y_true = df_p['Categories'].fillna('Non Haineux').astype(str)
        y_pred = df_p['prediction'].fillna('Non Haineux').astype(str)
        
        # On récupère la liste des catégories proprement
        labels_liste = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_liste)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_liste, yticklabels=labels_liste, cmap='Blues')
        plt.title("Matrice de Confusion : Analyse des erreurs de l'IA")
        plt.ylabel('Réalité (Humain)')
        plt.xlabel('Prédiction (IA)')
        plt.tight_layout()
        plt.savefig("matrice_confusion.png")
        print("✅ 3. Matrice de confusion : OK")
    else:
        print("❌ Fichier 'preds.csv' introuvable")

except Exception as e:
    print(f"⚠️ Erreur sur la matrice : {e}")
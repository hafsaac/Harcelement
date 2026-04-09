# SafeText — Détection automatique du harcèlement en ligne

> *Chaque jour, des millions de personnes subissent des formes de violence verbale en ligne ; insultes, menaces, discours de haine. Ces messages, souvent anonymes, laissent des traces profondes sur leurs victimes. Et pourtant, ils restent majoritairement invisibles, noyés dans le flux continu des réseaux sociaux.*
>
> *SafeText est né d'une conviction simple : la technologie peut ; et doit contribuer à rendre ces violences visibles, à les nommer, et à aider ceux qui les subissent.*

---

## Pourquoi ce projet ?

Le cyberharcèlement est un phénomène massif et sous-estimé. En France, **1 personne sur 3** a déjà été victime de harcèlement en ligne, et **78% des victimes** ne portent jamais plainte. Les formes de haine en ligne sont multiples, souvent subtiles, et difficiles à détecter manuellement à grande échelle.

Nous avons choisi ce sujet parce qu'il touche à des enjeux réels et urgents, la protection des personnes vulnérables, la modération des plateformes, et la responsabilité des algorithmes face aux discours de haine. C'est aussi un terrain d'application idéal pour la science des données : les textes, les émotions, les biais, les limites des modèles : tout est là.

---

## Ce que fait SafeText

SafeText est un outil d'intelligence artificielle capable de :

1. **Détecter** si un message est haineux ou non *(tâche binaire)*
2. **Identifier** la forme de haine parmi 6 catégories : Homophobie, Islamophobie, Racisme, Sexisme, Validisme, Xénophobie *(tâche multi-classes)*
3. **Expliquer** sa décision de manière pédagogique
4. **Transformer** un message haineux en version bienveillante
5. **Comparer** les performances de plusieurs modèles sur un même texte

Le tout est accessible via un site web interactif, conçu pour être utilisé par n'importe qui, pas seulement des data scientists.

---

## Les grandes étapes du projet

### 1. Constitution de la base de données
Nous avons assemblé et nettoyé une base de 2 640 messages en anglais, parfaitement équilibrée : 1 320 messages haineux et 1 320 messages neutres, répartis en 6 catégories de 220 messages chacune. Cet équilibre parfait était une condition essentielle pour entraîner des modèles fiables.

### 2. Analyse exploratoire (EDA)
Avant de construire le moindre modèle, nous avons passé plusieurs semaines à analyser nos données en profondeur ; vocabulaires dominants, profils émotionnels, longueurs des messages, niveaux d'agressivité stylistique. Cette étape n'était pas décorative : elle a directement orienté nos choix de modélisation.

### 3. Modélisation
Nous avons entraîné et comparé 5 approches différentes, du plus simple au plus complexe :
- **Régression Logistique** avec TF-IDF — rapide, interprétable, excellente baseline
- **TF-IDF classique** avec vectorisation personnalisée
- **Bag of Words** — approche par fréquences brutes
- **Random Forest** — modèle non linéaire pour capturer les interactions complexes
- **Toxic BERT** — modèle de langage pré-entraîné, capable de comprendre le contexte sémantique

### 4. Site web interactif
Nous avons déployé nos modèles sur un site web accessible en ligne, avec des fonctionnalités originales pensées pour rendre l'outil utile, pédagogique et engageant.

---

## Site web

**→ [http://217.160.27.219](http://217.160.27.219)**

Le site propose :
- **Analyse de messages** avec thermomètre de toxicité et ressources de prévention
- **Mode pédagogique** — visualisation des mots qui ont influencé la décision
- **Comparateur de modèles** — voir comment chaque algorithme réagit au même texte
- **Traducteur de Bienveillance** — transformer un message haineux en formulation constructive
- **Quiz IA** — deviner la catégorie de harcèlement en jouant contre notre modèle
- **Page Visualisations** — les graphiques clés de notre analyse exploratoire
- **Page Prévention** — chiffres, ressources et contacts utiles

---

## Structure du dépôt

```
📦 Harcelement/
│
├── 📂 Archives/
│   └── Contient les versions antérieures du projet : anciens modèles
│       conceptuels (MC/DMD), anciennes versions des comptes rendus,
│       fichiers de travail intermédiaires. Ces fichiers témoignent
│       de l'évolution de notre démarche au fil des semaines.
│
├── 📂 Base de données/
│   ├── Harcelement.csv          ← base complète (2 640 messages)
│   ├── Base_Train2.csv          ← jeu d'entraînement (2 400 messages)
│   └── Base_Test2.csv           ← jeu de test (240 messages)
│
├── 📂 Code/
│   ├── Notebook_Modele_IA.ipynb ← notebook principal — EDA + 5 modèles
│   └── *.pkl                    ← modèles entraînés sauvegardés (joblib)
│
├── 📂 Compte rendu/
│   ├── CR_semaine_*.pdf         ← comptes rendus hebdomadaires (versions finales)
│   └── 📂 Archives/             ← anciennes versions des comptes rendus
│
├── 📂 Modeles conceptuels/
│   ├── MC_*.png / DMD_*.png     ← modèles conceptuels et diagrammes
│   └── Harcelement.sql          ← structure de la base de données
│
├── 📂 Site web/
│   ├── index.php                ← page principale (analyse+thermomètre)
│   ├── quiz.php                 ← mode quiz (joueur vs IA)
│   ├── visualisations.php       ← visualisations clés de l'EDA
│   ├── traducteur-page.php      ← traducteur de bienveillance
│   ├── prevention.php           ← ressources et chiffres clés
│   ├── predict.py / predict.php ← moteur de prédiction
│   ├── comparer.py / comparer.php ← comparateur de modèles
│   ├── traducteur.py / traducteur.php ← reformulation bienveillante
│   ├── 📂 styles/               ← feuille de style CSS
│   ├── 📂 models/               ← fichiers .pkl des modèles
│   └── 📂 Images/               ← visuels utilisés sur le site
│
└── 📂 Visualisations/
    └── visu_final.pdf           ← rapport complet des 14 visualisations EDA
```

---

## Équipe

Projet réalisé dans le cadre de l'UE **Science des Données 4** — L3 MIASHS

**Chloé · Sarah · Hafsa · Nadir**

Université Paul Valery, Montpellier — 2025/2026

---

## Technologies utilisées

| Catégorie | Outils |
|-----------|--------|
| Langage principal | Python 3.12 |
| Machine Learning | scikit-learn, transformers (HuggingFace) |
| Modèle BERT | unitary/toxic-bert |
| Traitement texte | TF-IDF, Bag of Words, joblib |
| Analyse de données | pandas, numpy, matplotlib, seaborn |
| Analyse émotionnelle | Lexique NRC |
| Site web | PHP, Python (scripts), Apache |
| Hébergement | IONOS VPS — Ubuntu 24.04 |
| Versioning | Git / GitHub |

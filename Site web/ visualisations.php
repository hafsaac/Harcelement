<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Visualisations</title>
  <link rel="stylesheet" href="styles/style.css">
  <style>
    .visu-section {
      margin-bottom: 60px;
    }

    .visu-img {
      width: 100%;
      border-radius: var(--radius);
      border: 1px solid var(--color-border);
      margin: 24px 0;
      box-shadow: var(--shadow);
    }

    .visu-legende {
      background: rgba(102,126,234,0.06);
      border-left: 3px solid var(--color-purple);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      padding: 16px 20px;
      color: var(--color-muted);
      font-size: 0.92rem;
      line-height: 1.7;
      margin-bottom: 16px;
    }

    .visu-legende strong {
      color: var(--color-text);
      display: block;
      margin-bottom: 6px;
      font-size: 1rem;
    }

    .visu-apport {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      padding: 16px 20px;
      margin-top: 16px;
    }

    .visu-apport-title {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--color-purple);
      margin-bottom: 8px;
    }

    .visu-apport p {
      color: var(--color-muted);
      font-size: 0.9rem;
      line-height: 1.7;
      margin: 0;
    }

    .tag-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }

    .tag {
      padding: 4px 14px;
      border-radius: 50px;
      font-size: 0.8rem;
      font-weight: 600;
      border: 1px solid;
    }

    .tag-purple { 
      background: rgba(102,126,234,0.1); 
      border-color: rgba(102,126,234,0.4); 
      color: #667eea; 
    }

    .tag-pink { 
      background: rgba(245,87,108,0.1); 
      border-color: rgba(245,87,108,0.4); 
      color: #f5576c; 
    }

    .tag-cyan { 
      background: rgba(0,242,254,0.1); 
      border-color: rgba(0,242,254,0.4); 
      color: #00f2fe; 
    }

    .tag-orange { 
      background: rgba(251,146,60,0.1); 
      border-color: rgba(251,146,60,0.4); 
      color: #fb923c; 
    }

    .numero-visu {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: var(--gradient-main);
      color: white;
      font-weight: 800;
      font-size: 0.9rem;
      margin-right: 12px;
      flex-shrink: 0;
    }

    .visu-header {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
    }

    .visu-header h2 {
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--color-text);
    }

    .intro-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }

    .intro-stat {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      padding: 20px;
      text-align: center;
    }

    .intro-stat-num {
      font-size: 2rem;
      font-weight: 800;
      background: var(--gradient-main);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .intro-stat-label {
      font-size: 0.82rem;
      color: var(--color-muted);
      margin-top: 4px;
      line-height: 1.4;
    }
  </style>
</head>
<body>

<!-- ===================== NAVIGATION ===================== -->
<nav>
  <a href="index.php" class="nav-logo">🛡️ SafeText</a>
  <ul class="nav-links">
    <li><a href="index.php">Analyser</a></li>
    <li><a href="quiz.php">Quiz</a></li>
    <li><a href="visualisations.php" class="active">Visualisations</a></li>
    <li><a href="prevention.php">Prévention</a></li>
  </ul>
</nav>

<main>

  <!-- HERO -->
  <section class="hero">
    <h1>Ce que les données<br>nous ont appris</h1>
    <p>Avant de construire nos modèles, nous avons analysé en profondeur
       nos 2 640 messages. Voici les visualisations les plus stratégiques
       et ce qu'elles ont orienté dans nos choix de modélisation.</p>
  </section>

  <!-- CHIFFRES INTRO -->
  <div class="intro-stats">
    <div class="intro-stat">
      <div class="intro-stat-num">2 640</div>
      <div class="intro-stat-label">messages analysés au total</div>
    </div>
    <div class="intro-stat">
      <div class="intro-stat-num">50/50</div>
      <div class="intro-stat-label">équilibre parfait haineux / non haineux</div>
    </div>
    <div class="intro-stat">
      <div class="intro-stat-num">6</div>
      <div class="intro-stat-label">catégories de harcèlement, 220 messages chacune</div>
    </div>
    <div class="intro-stat">
      <div class="intro-stat-num">14</div>
      <div class="intro-stat-label">visualisations produites lors de l'EDA</div>
    </div>
  </div>

  <!-- ====== VISU 1 ====== -->
  <div class="card visu-section">
    <div class="visu-header">
      <span class="numero-visu">1</span>
      <h2>Nuage comparatif — Haineux vs Non Haineux</h2>
    </div>

    <div class="tag-grid">
      <span class="tag tag-purple">Analyse lexicale</span>
      <span class="tag tag-pink">Tâche binaire</span>
      <span class="tag tag-cyan">Séparabilité des classes</span>
    </div>

    <img src="images/nuage_comparatif.png" alt="Nuage comparatif Haineux vs Non Haineux" class="visu-img">

    <div class="visu-legende">
      <strong>📖 Lecture du graphique</strong>
      La séparation visuelle est immédiate et frappante. Les grands mots violets (haineux) ciblent
      directement des groupes sociaux ou expriment de la violence verbale. Les grands mots bleus
      (neutres) reflètent des thèmes anodins du quotidien — créativité, cuisine, encouragement.
    </div>

    <div class="visu-apport">
      <div class="visu-apport-title">🎯 Ce que ça nous a apporté pour la modélisation</div>
      <p>Cette divergence lexicale confirme que les deux classes sont <strong style="color:var(--color-text)">linéairement séparables dans l'espace TF-IDF</strong>.
      Cette visualisation a justifié notre choix de commencer par des modèles linéaires simples
      (Régression Logistique, TF-IDF) avant d'aller vers des architectures plus complexes.
      Un SVM linéaire ou une régression logistique peuvent atteindre d'excellentes performances
      sur cette tâche binaire sans nécessiter du deep learning.</p>
    </div>
  </div>

  <!-- ====== VISU 2 ====== -->
  <div class="card visu-section">
    <div class="visu-header">
      <span class="numero-visu">2</span>
      <h2>Top 10 des mots par catégorie de harcèlement</h2>
    </div>

    <div class="tag-grid">
      <span class="tag tag-purple">Analyse multi-classes</span>
      <span class="tag tag-orange">Feature engineering</span>
      <span class="tag tag-pink">Zones d'ambiguïté</span>
    </div>

    <img src="images/top10_mots.png" alt="Top 10 mots par catégorie" class="visu-img">

    <div class="visu-legende">
      <strong>📖 Lecture du graphique</strong>
      Chaque catégorie révèle un vocabulaire central distinctif : "gay", "queer" pour l'Homophobie ;
      "muslims", "camel" pour l'Islamophobie ; "black", "nigger" pour le Racisme ; "women", "bitch"
      pour le Sexisme ; "disabled" pour le Validisme ; "immigrants" pour la Xénophobie.
      Mais des termes comme "typical", "nothing", "lives", "country" apparaissent dans plusieurs catégories.
    </div>

    <div class="visu-apport">
      <div class="visu-apport-title">🎯 Ce que ça nous a apporté pour la modélisation</div>
      <p>La présence de vocabulaire partagé entre catégories a orienté deux décisions clés.
      D'abord, l'approche TF-IDF sera suffisante grâce aux marqueurs exclusifs par catégorie.
      Ensuite, les paires Racisme/Xénophobie et Islamophobie/Xénophobie seront les plus difficiles
      à distinguer — ce que nos matrices de confusion ont confirmé par la suite.
      C'est aussi ce chevauchement qui justifie l'utilisation de <strong style="color:var(--color-text)">Toxic BERT</strong>
      pour les cas ambigus que les modèles classiques ne savent pas gérer.</p>
    </div>
  </div>

  <!-- ====== VISU 3 ====== -->
  <div class="card visu-section">
    <div class="visu-header">
      <span class="numero-visu">3</span>
      <h2>Profil émotionnel NRC par catégorie</h2>
    </div>

    <div class="tag-grid">
      <span class="tag tag-cyan">Analyse émotionnelle</span>
      <span class="tag tag-purple">Lexique NRC</span>
      <span class="tag tag-orange">Features numériques</span>
    </div>

    <img src="images/profil_nrc.png" alt="Profil émotionnel NRC par catégorie" class="visu-img">

    <div class="visu-legende">
      <strong>📖 Lecture du graphique</strong>
      Chaque catégorie présente une signature émotionnelle distincte : la colère domine dans
      l'Islamophobie et le Sexisme, la tristesse ressort dans le Racisme, la peur caractérise
      la Xénophobie et le Validisme. Ces nuances émotionnelles sont invisibles dans une simple
      analyse de fréquences de mots.
    </div>

    <div class="visu-apport">
      <div class="visu-apport-title">🎯 Ce que ça nous a apporté pour la modélisation</div>
      <p>Les scores NRC constituent des <strong style="color:var(--color-text)">features numériques complémentaires au TF-IDF</strong>.
      Dans un modèle à features mixtes (Random Forest, Gradient Boosting), ces 8 dimensions
      émotionnelles enrichissent la représentation de chaque message et améliorent la robustesse
      sur les cas difficiles — notamment les messages qui utilisent un vocabulaire non haineux
      mais véhiculent des émotions de colère ou de dégoût caractéristiques du discours haineux.</p>
    </div>
  </div>

  <!-- ====== VISU 4 ====== -->
  <div class="card visu-section">
    <div class="visu-header">
      <span class="numero-visu">4</span>
      <h2>Répartition des 6 catégories de harcèlement</h2>
    </div>

    <div class="tag-grid">
      <span class="tag tag-purple">Équilibre des classes</span>
      <span class="tag tag-cyan">Qualité des données</span>
      <span class="tag tag-pink">Métriques d'évaluation</span>
    </div>

    <img src="images/repartition_categories.png" alt="Répartition des 6 catégories" class="visu-img">

    <div class="visu-legende">
      <strong>📖 Lecture du graphique</strong>
      Chaque catégorie contient exactement 220 messages — un équilibre parfait entre les
      6 formes de harcèlement. Cette uniformité est une garantie fondamentale : le modèle
      ne pourra pas "tricher" en favorisant les catégories les plus fréquentes.
    </div>

    <div class="visu-apport">
      <div class="visu-apport-title">🎯 Ce que ça nous a apporté pour la modélisation</div>
      <p>Un dataset parfaitement équilibré nous a permis d'utiliser directement
      <strong style="color:var(--color-text)">l'accuracy et le F1-score macro comme métriques principales</strong>
      sans recourir à des techniques de rééquilibrage (SMOTE, undersampling, pondération des classes).
      Cela simplifie le pipeline et garantit que toutes les catégories sont traitées avec
      la même importance lors de l'entraînement et de l'évaluation.</p>
    </div>
  </div>

  <!-- CONCLUSION -->
  <div class="card" style="text-align:center; padding: 40px;">
    <div class="card-title" style="justify-content:center;">🔬 Conclusion de l'analyse exploratoire</div>
    <p style="color:var(--color-muted); max-width:700px; margin:0 auto 24px; line-height:1.8;">
      Ces visualisations nous ont permis de construire une stratégie de modélisation informée :
      commencer par des modèles linéaires simples sur les features TF-IDF, enrichir avec des
      features numériques (longueur, scores NRC, agressivité stylistique), et réserver
      Toxic BERT pour les cas ambigus que les modèles classiques ne savent pas résoudre.
    </p>
    <a href="index.php" style="
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 12px 28px;
      background: var(--gradient-main);
      color: white;
      border-radius: 50px;
      text-decoration: none;
      font-weight: 700;
      font-size: 0.95rem;
    ">🔍 Tester le modèle</a>
  </div>

</main>

</body>
</html>
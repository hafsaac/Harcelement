<?php
$compteur_file = 'compteur.txt';
$compteur = file_exists($compteur_file) ? (int)file_get_contents($compteur_file) : 0;
?>
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Détecteur de harcèlement</title>
  <link rel="stylesheet" href="styles/style.css">
</head>
<body>

<nav>
  <a href="index.php" class="nav-logo">🛡️ SafeText</a>
  <ul class="nav-links">
    <li><a href="index.php" class="active">Analyser</a></li>
    <li><a href="quiz.php">Quiz</a></li>
    <li><a href="visualisations.php">Visualisations</a></li>
    <li><a href="prevention.php">Prévention</a></li>
  </ul>
</nav>

<!-- AVATAR EMPATHIQUE -->
<div id="avatar-container" style="position:fixed; bottom:30px; right:30px; z-index:999; text-align:center;">
  <div id="avatar-bulle" style="background:var(--color-card); border:1px solid var(--color-border); border-radius:12px; padding:10px 14px; font-size:0.82rem; color:var(--color-muted); margin-bottom:8px; max-width:180px; display:none; line-height:1.5;"></div>
  <svg id="avatar-svg" width="80" height="80" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="35" fill="#1a1a2e" stroke="#667eea" stroke-width="2"/>
    <circle id="oeil-gauche" cx="28" cy="35" r="5" fill="#667eea"/>
    <circle id="oeil-droit" cx="52" cy="35" r="5" fill="#667eea"/>
    <circle id="pupille-gauche" cx="29" cy="36" r="2.5" fill="white"/>
    <circle id="pupille-droite" cx="53" cy="36" r="2.5" fill="white"/>
    <path id="bouche" d="M 28 52 Q 40 62 52 52" stroke="#667eea" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path id="sourcil-gauche" d="M 22 27 Q 28 24 34 27" stroke="#667eea" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path id="sourcil-droit" d="M 46 27 Q 52 24 58 27" stroke="#667eea" stroke-width="2" fill="none" stroke-linecap="round"/>
  </svg>
  <div style="font-size:0.72rem; color:var(--color-muted); margin-top:4px;">SafeBot</div>
</div>

<main>

  <section class="hero">
    <h1>Détectez le harcèlement<br>en un instant</h1>
    <p>Un outil d'intelligence artificielle pour identifier et comprendre
       les formes de discours haineux en ligne.</p>
    <div class="counter-badge">
      <span class="counter-dot"></span>
      <span id="compteur-val"><?= number_format($compteur, 0, ',', ' ') ?></span>
      messages analysés
    </div>
  </section>

  <div class="card">
    <div class="card-title">Analyser un message</div>
    <div class="form-area">
      <textarea id="texte-input" placeholder="Collez ou tapez un message à analyser..." maxlength="1000"></textarea>
      <div class="char-count"><span id="char-count">0</span> / 1000</div>
      <button class="btn btn-primary" id="btn-analyser" onclick="analyser()">
        <span id="btn-text">Analyser le message</span>
      </button>
    </div>
  </div>

  <div class="card" id="resultat">
  <div class="card-title">Résultat de l'analyse</div>

  <div id="result-texte-cite" style="font-style:italic; color:var(--color-muted); margin-bottom:20px; font-size:0.95rem; border-left:3px solid var(--color-border); padding-left:14px;"></div>

  <!-- THERMOMÈTRE DE TOXICITÉ -->
  <div style="margin-bottom:24px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <span style="font-size:0.85rem; color:var(--color-muted);">Niveau de toxicité</span>
      <span id="thermo-valeur" style="font-weight:700; font-size:0.9rem;"></span>
    </div>
    <div style="height:14px; background:rgba(255,255,255,0.06); border-radius:50px; overflow:hidden;">
      <div id="thermo-barre" style="height:100%; width:0%; border-radius:50px; transition: width 1s ease, background 1s ease;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:4px;">
      <span style="font-size:0.72rem; color:#00f2fe;">Sain</span>
      <span style="font-size:0.72rem; color:#f59e0b;">Modéré</span>
      <span style="font-size:0.72rem; color:#f5576c;">Toxique</span>
    </div>
  </div>

  <div class="result-label" id="result-label"></div>
  <div class="result-confiance" id="result-confiance"></div>
  <div id="result-categorie"></div>
  <div class="prevention-box" id="result-prevention" style="display:none"></div>

  <!-- MODE PÉDAGOGIQUE -->
  <div id="mode-peda" style="display:none; margin-top:20px;">
    <div style="border-top:1px solid var(--color-border); padding-top:20px;">
      <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--color-purple); margin-bottom:12px;">
        Mode pédagogique — Comment le modèle a décidé
      </div>
      <div id="peda-mots" style="line-height:2; font-size:0.95rem;"></div>
      <div style="margin-top:12px; padding:12px 16px; background:rgba(102,126,234,0.06); border-radius:var(--radius-sm); font-size:0.85rem; color:var(--color-muted);">
        Les mots <mark style="background:rgba(245,87,108,0.3); color:var(--color-text); padding:2px 6px; border-radius:4px;">surlignés en rouge</mark> ont le plus contribué à la décision du modèle.
        Plus la couleur est intense, plus le mot est discriminant.
      </div>
    </div>
  </div>

  <button id="btn-peda" onclick="togglePeda()" style="
    display:none;
    margin-top:16px;
    padding:10px 20px;
    border-radius:50px;
    border:1px solid var(--color-purple);
    background:rgba(102,126,234,0.1);
    color:var(--color-purple);
    font-family:'Inter',sans-serif;
    font-size:0.85rem;
    font-weight:600;
    cursor:pointer;
    transition:all 0.3s ease;
  ">Voir comment le modèle a décidé</button>

  <!-- COMPARATEUR DE MODÈLES -->
  <div id="comparateur" style="display:none; margin-top:24px;">
    <div style="border-top:1px solid var(--color-border); padding-top:20px;">
      <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--color-purple); margin-bottom:16px;">
        Comparateur — Que disent les autres modèles ?
      </div>
      <div id="comparateur-contenu"></div>
    </div>
  </div>

  <button id="btn-comparer" onclick="comparer()" style="
    display:none;
    margin-top:12px;
    padding:10px 20px;
    border-radius:50px;
    border:1px solid rgba(255,255,255,0.15);
    background:rgba(255,255,255,0.04);
    color:var(--color-muted);
    font-family:'Inter',sans-serif;
    font-size:0.85rem;
    font-weight:600;
    cursor:pointer;
    transition:all 0.3s ease;
  ">Comparer avec les autres modèles</button>
</div>

  <div class="card">
    <div class="card-title">Les 6 formes détectées</div>
    <div class="categories-grid">
      <div class="cat-card">
        <div class="cat-icon">🏳️‍🌈</div>
        <div class="cat-name">Homophobie</div>
        <div class="cat-desc">Discriminations envers les personnes LGBTQ+</div>
      </div>
      <div class="cat-card">
        <div class="cat-icon">🕌</div>
        <div class="cat-name">Islamophobie</div>
        <div class="cat-desc">Haine envers les musulmans ou l'islam</div>
      </div>
      <div class="cat-card">
        <div class="cat-icon">✊</div>
        <div class="cat-name">Racisme</div>
        <div class="cat-desc">Discriminations fondées sur l'origine ou la couleur de peau</div>
      </div>
      <div class="cat-card">
        <div class="cat-icon">⚧️</div>
        <div class="cat-name">Sexisme</div>
        <div class="cat-desc">Discriminations fondées sur le genre</div>
      </div>
      <div class="cat-card">
        <div class="cat-icon">♿</div>
        <div class="cat-name">Validisme</div>
        <div class="cat-desc">Discriminations envers les personnes handicapées</div>
      </div>
      <div class="cat-card">
        <div class="cat-icon">🌍</div>
        <div class="cat-name">Xénophobie</div>
        <div class="cat-desc">Haine envers les étrangers ou immigrés</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Performances de nos modèles</div>
    <div id="perf-bars">
      <div class="perf-bar">
        <div class="perf-label">Toxic BERT</div>
        <div class="perf-track"><div class="perf-fill" data-value="94"></div></div>
        <div class="perf-value">94%</div>
      </div>
      <div class="perf-bar">
        <div class="perf-label">TF-IDF</div>
        <div class="perf-track"><div class="perf-fill" data-value="97"></div></div>
        <div class="perf-value">97%</div>
      </div>
      <div class="perf-bar">
        <div class="perf-label">Régression Logistique</div>
        <div class="perf-track"><div class="perf-fill" data-value="89"></div></div>
        <div class="perf-value">89%</div>
      </div>
      <div class="perf-bar">
        <div class="perf-label">Random Forest</div>
        <div class="perf-track"><div class="perf-fill" data-value="88"></div></div>
        <div class="perf-value">88%</div>
      </div>
      <div class="perf-bar">
        <div class="perf-label">Bag of Words</div>
        <div class="perf-track"><div class="perf-fill" data-value="88"></div></div>
        <div class="perf-value">88%</div>
      </div>
    </div>
  </div>

</main>

<script>

// =========================================================
// DONNÉES DE PRÉVENTION
// =========================================================
const prevention = {
  "Homophobie": { emoji: "🏳️‍🌈", couleur: "#a78bfa", message: "L'homophobie est une discrimination illégale en France. Elle peut causer des dommages psychologiques profonds. Si vous êtes victime, vous pouvez contacter SOS Homophobie au 01 48 06 42 41." },
  "Islamophobie": { emoji: "🕌", couleur: "#34d399", message: "L'islamophobie est un délit en France. Tout acte discriminatoire basé sur la religion peut être signalé auprès des autorités. Le CCIF accompagne les victimes de discriminations islamophobes." },
  "Racisme": { emoji: "✊", couleur: "#f59e0b", message: "Le racisme est un crime en France, passible de peines d'emprisonnement. Signalez tout contenu raciste sur la plateforme Pharos. Des associations comme SOS Racisme peuvent vous accompagner." },
  "Sexisme": { emoji: "⚧️", couleur: "#f472b6", message: "Le sexisme, notamment le harcèlement sexiste en ligne, est sanctionné par la loi. Vous pouvez signaler les contenus sur les plateformes et contacter le 3919 (violences femmes info)." },
  "Validisme": { emoji: "♿", couleur: "#60a5fa", message: "Le validisme — discrimination envers les personnes handicapées — est interdit par la loi. L'APF France Handicap et le Défenseur des droits peuvent vous aider en cas de discrimination." },
  "Xénophobie": { emoji: "🌍", couleur: "#fb923c", message: "La xénophobie est un délit en France. Tout discours haineux envers des étrangers peut être signalé sur Pharos. La LICRA et la LDH accompagnent les victimes de discriminations." }
};

// =========================================================
// AVATAR EMPATHIQUE
// =========================================================
const avatarEtats = {
  neutre:   { bouche: "M 28 52 Q 40 62 52 52", sourcilG: "M 22 27 Q 28 24 34 27", sourcilD: "M 46 27 Q 52 24 58 27", couleur: "#667eea", bulle: "" },
  heureux:  { bouche: "M 24 50 Q 40 66 56 50", sourcilG: "M 22 26 Q 28 22 34 26", sourcilD: "M 46 26 Q 52 22 58 26", couleur: "#00f2fe", bulle: "Ce message est positif !" },
  triste:   { bouche: "M 28 58 Q 40 48 52 58", sourcilG: "M 22 28 Q 28 32 34 28", sourcilD: "M 46 28 Q 52 32 58 28", couleur: "#f5576c", bulle: "Ce message est blessant..." },
  choque:   { bouche: "M 33 56 Q 40 62 47 56", sourcilG: "M 22 30 Q 28 24 34 30", sourcilD: "M 46 30 Q 52 24 58 30", couleur: "#f59e0b", bulle: "Discours haineux détecté" },
  reflechi: { bouche: "M 28 53 Q 40 57 52 53", sourcilG: "M 22 27 Q 28 23 34 27", sourcilD: "M 46 25 Q 52 27 58 25", couleur: "#a78bfa", bulle: "Analyse en cours..." }
};

function changerAvatar(etat) {
  const e = avatarEtats[etat];
  document.getElementById('bouche').setAttribute('d', e.bouche);
  document.getElementById('sourcil-gauche').setAttribute('d', e.sourcilG);
  document.getElementById('sourcil-droit').setAttribute('d', e.sourcilD);
  document.getElementById('oeil-gauche').setAttribute('fill', e.couleur);
  document.getElementById('oeil-droit').setAttribute('fill', e.couleur);
  const bulle = document.getElementById('avatar-bulle');
  if (e.bulle) {
    bulle.textContent = e.bulle;
    bulle.style.display = 'block';
    setTimeout(() => { bulle.style.display = 'none'; }, 3000);
  }
}

// =========================================================
// GESTION DU TEXTAREA (un seul listener)
// =========================================================
document.getElementById('texte-input').addEventListener('input', function() {
  document.getElementById('char-count').textContent = this.value.length;
  if (this.value.length > 10) {
    changerAvatar('reflechi');
  } else {
    changerAvatar('neutre');
  }
});

// =========================================================
// ANALYSE PRINCIPALE
// =========================================================
async function analyser() {
  const texte = document.getElementById('texte-input').value.trim();

  if (!texte) {
    alert('Veuillez entrer un message à analyser.');
    return;
  }

  changerAvatar('reflechi');
  const btn = document.getElementById('btn-analyser');
  const btnText = document.getElementById('btn-text');
  btn.disabled = true;
  btnText.innerHTML = '<div class="spinner"></div> Analyse en cours...';

  try {
    const formData = new FormData();
    formData.append('texte', texte);
    const response = await fetch('predict.php', { method: 'POST', body: formData });
    const data = await response.json();

    if (data.erreur) {
      changerAvatar('triste');
      alert('Erreur : ' + data.erreur);
      return;
    }

    const estHaineux = data.binaire === 'Haineux' || data.binaire === '1';
    changerAvatar(estHaineux ? 'choque' : 'heureux');
    afficherResultat(data, texte);

  } catch (e) {
    changerAvatar('triste');
    alert('Erreur de connexion.');
  } finally {
    btn.disabled = false;
    btnText.innerHTML = 'Analyser un autre message';
  }
}

// =========================================================
// MOTS CLÉS PAR CATÉGORIE (pour mode pédagogique)
// =========================================================
const motsCles = {
  "Homophobie":    ["gay","queer","faggot","fag","lesbian","homosexual","dyke"],
  "Islamophobie":  ["muslim","muslims","islam","jihad","terrorist","camel","mosque"],
  "Racisme":       ["black","nigger","nigga","negro","coon","racial","slave"],
  "Sexisme":       ["women","woman","female","bitch","slut","whore","gender"],
  "Validisme":     ["disabled","cripple","retard","wheelchair","handicap","mong"],
  "Xénophobie":    ["immigrant","immigrants","foreign","refugee","border","invasion"],
  "Haineux":       ["hate","kill","die","disgusting","worthless","stupid","scum","trash"]
};

const motsNégatifs = ["hate","kill","die","disgusting","worthless","stupid","scum","never","nothing","typical","must","deserve","suffer"];

// =========================================================
// AFFICHAGE DU RÉSULTAT ENRICHI
// =========================================================
function afficherResultat(data, texte) {
  const bloc = document.getElementById('resultat');
  const label = document.getElementById('result-label');
  const confiance = document.getElementById('result-confiance');
  const categorieDiv = document.getElementById('result-categorie');
  const preventionBox = document.getElementById('result-prevention');
  const texteCite = document.getElementById('result-texte-cite');

  const extrait = texte.length > 100 ? texte.substring(0, 100) + '…' : texte;
  texteCite.textContent = '« ' + extrait + ' »';

  const estHaineux = data.binaire === 'Haineux' || data.binaire === '1';
  const confVal = parseFloat(data.confiance_bin);

  // --- THERMOMÈTRE ---
  const toxicite = estHaineux ? confVal : (100 - confVal);
  const barre = document.getElementById('thermo-barre');
  const valeur = document.getElementById('thermo-valeur');

  setTimeout(() => { barre.style.width = toxicite + '%'; }, 100);

  let couleur, label_thermo;
  if (toxicite < 40) {
    couleur = 'linear-gradient(90deg, #00f2fe, #4facfe)';
    label_thermo = 'Faible';
  } else if (toxicite < 70) {
    couleur = 'linear-gradient(90deg, #f59e0b, #fb923c)';
    label_thermo = 'Modéré';
  } else {
    couleur = 'linear-gradient(90deg, #f5576c, #f093fb)';
    label_thermo = 'Élevé';
  }
  barre.style.background = couleur;
  valeur.textContent = Math.round(toxicite) + '% — ' + label_thermo;

  // --- LABEL ---
  if (estHaineux) {
    label.textContent = 'Message haineux';
    label.className = 'result-label haineux';
    bloc.className = 'card result-haineux';
  } else {
    label.textContent = 'Message non haineux';
    label.className = 'result-label safe';
    bloc.className = 'card result-safe';
  }

  confiance.textContent = 'Confiance du modèle : ' + data.confiance_bin + '%';

  // --- CATÉGORIE ---
  if (estHaineux && data.categorie && data.categorie !== 'Non applicable') {
    const info = prevention[data.categorie] || { emoji: '⚠️', couleur: '#f5576c' };
    categorieDiv.innerHTML = `
      <div class="categorie-badge" style="background:${info.couleur}20; border:1px solid ${info.couleur}60; color:${info.couleur}; margin-bottom:16px;">
        ${info.emoji} ${data.categorie}
        <span style="font-weight:400; color:var(--color-muted); font-size:0.82rem;">— confiance ${data.confiance_cat}%</span>
      </div>`;
    preventionBox.innerHTML = `<strong>Information & ressources</strong>${info.message}`;
    preventionBox.style.display = 'block';
  } else {
    categorieDiv.innerHTML = '';
    preventionBox.style.display = 'none';
  }

  // --- MODE PÉDAGOGIQUE ---
  const categorie = data.categorie || 'Haineux';
  const cles = [...(motsCles[categorie] || []), ...motsNégatifs];
  const mots = texte.split(' ');
  const motsHtml = mots.map(mot => {
    const motNet = mot.toLowerCase().replace(/[^a-z]/g, '');
    const match = cles.find(c => motNet.includes(c) || c.includes(motNet));
    if (match && motNet.length > 2) {
      return `<mark style="background:rgba(245,87,108,0.35); color:var(--color-text); padding:2px 6px; border-radius:4px; margin:2px;">${mot}</mark>`;
    }
    return `<span style="margin:2px;">${mot}</span>`;
  }).join(' ');
  document.getElementById('peda-mots').innerHTML = motsHtml;
  document.getElementById('btn-peda').style.display = 'inline-block';
  document.getElementById('btn-comparer').style.display = 'inline-block';
  document.getElementById('comparateur').style.display = 'none';
  document.getElementById('mode-peda').style.display = 'none';

  bloc.style.display = 'block';
  bloc.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// =========================================================
// MODE PÉDAGOGIQUE
// =========================================================
let pedaOuvert = false;
function togglePeda() {
  pedaOuvert = !pedaOuvert;
  document.getElementById('mode-peda').style.display = pedaOuvert ? 'block' : 'none';
  document.getElementById('btn-peda').textContent = pedaOuvert
    ? 'Masquer l\'explication'
    : 'Voir comment le modèle a décidé';
}

// =========================================================
// COMPARATEUR DE MODÈLES
// =========================================================
async function comparer() {
  const texte = document.getElementById('texte-input').value.trim();
  const btn = document.getElementById('btn-comparer');
  btn.textContent = 'Comparaison en cours...';
  btn.disabled = true;

  try {
    const fd = new FormData();
    fd.append('texte', texte);
    const resp = await fetch('comparer.php', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.erreur) { alert('Erreur : ' + data.erreur); return; }

    let html = '<div style="display:grid; gap:10px;">';
    data.resultats.forEach(r => {
      const estH = r.binaire === 'Haineux';
      const couleur = estH ? '#f5576c' : '#00f2fe';
      html += `
        <div style="display:flex; align-items:center; gap:12px; padding:12px 16px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm); border:1px solid var(--color-border);">
          <div style="width:140px; font-size:0.85rem; font-weight:600; color:var(--color-text);">${r.modele}</div>
          <div style="padding:3px 12px; border-radius:50px; font-size:0.8rem; font-weight:700; background:${couleur}20; border:1px solid ${couleur}60; color:${couleur};">${r.binaire}</div>
          <div style="font-size:0.8rem; color:var(--color-muted); flex:1;">${r.categorie !== 'Non applicable' ? '→ ' + r.categorie : ''}</div>
          <div style="font-size:0.8rem; font-weight:700; color:var(--color-text);">${r.confiance}%</div>
        </div>`;
    });
    html += '</div>';

    document.getElementById('comparateur-contenu').innerHTML = html;
    document.getElementById('comparateur').style.display = 'block';
    document.getElementById('comparateur').scrollIntoView({ behavior: 'smooth' });

  } catch(e) {
    alert('Erreur de connexion.');
  } finally {
    btn.textContent = 'Comparer avec les autres modèles';
    btn.disabled = false;
  }
}
// =========================================================
// BARRES DE PERFORMANCE
// =========================================================
function animerBarres() {
  document.querySelectorAll('.perf-fill').forEach(bar => {
    const val = bar.getAttribute('data-value');
    setTimeout(() => { bar.style.width = val + '%'; }, 300);
  });
}

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) { animerBarres(); observer.disconnect(); }
  });
}, { threshold: 0.3 });

observer.observe(document.getElementById('perf-bars'));

document.getElementById('texte-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && e.ctrlKey) analyser();
});

</script>
</body>
</html>

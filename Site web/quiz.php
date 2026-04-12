<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Quiz</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>

<!-- ===================== NAVIGATION ===================== -->
<nav>
  <a href="index.php" class="nav-logo">🛡️ SafeText</a>
  <ul class="nav-links">
    <li><a href="index.php">Analyser</a></li>
    <li><a href="quiz.php" class="active">Quiz</a></li>
    <li><a href="prevention.php">Prévention</a></li>
  </ul>
</nav>

<main>

  <!-- HERO -->
  <section class="hero">
    <h1>Mode Quiz</h1>
    <p>Un message s'affiche — à toi de deviner de quelle forme de harcèlement il s'agit.
       L'IA joue en même temps que toi. Qui a le meilleur score ?</p>
  </section>

  <!-- SCORE -->
  <div class="card" style="text-align:center; padding: 20px 32px;">
    <div style="display:flex; justify-content:center; gap:48px; flex-wrap:wrap;">
      <div>
        <div class="score-display" id="score-toi">0</div>
        <div style="color:var(--color-muted); font-size:0.85rem; margin-top:4px;">Ton score</div>
      </div>
      <div style="display:flex; align-items:center; color:var(--color-muted); font-size:1.5rem;">VS</div>
      <div>
        <div class="score-display" id="score-ia">0</div>
        <div style="color:var(--color-muted); font-size:0.85rem; margin-top:4px;">Score IA</div>
      </div>
    </div>
    <div style="margin-top:12px; color:var(--color-muted); font-size:0.85rem;">
      Question <span id="q-num">1</span> / <span id="q-total">10</span>
    </div>
  </div>

  <!-- QUESTION -->
  <div class="card" id="quiz-zone">
    <div class="card-title">💬 Quel type de harcèlement est-ce ?</div>

    <div class="quiz-message" id="quiz-message">
      Chargement...
    </div>

    <div class="quiz-choices" id="quiz-choices"></div>

    <div id="quiz-feedback" style="display:none; margin-top:20px;">
      <div class="prevention-box">
        <strong id="feedback-titre"></strong>
        <span id="feedback-texte"></span>
      </div>
      <button class="btn btn-primary" id="btn-suivant" onclick="questionSuivante()" style="margin-top:16px;">
        ➡️ Question suivante
      </button>
    </div>
  </div>

  <!-- RÉSULTAT FINAL -->
  <div class="card" id="quiz-fin" style="display:none; text-align:center;">
    <div class="card-title" style="justify-content:center;">🏆 Résultat final</div>
    <div class="score-display" id="score-final"></div>
    <p style="color:var(--color-muted); margin-top:12px; margin-bottom:28px;" id="message-final"></p>
    <button class="btn btn-primary" onclick="relancerQuiz()" style="max-width:300px; margin:0 auto;">
      🔄 Rejouer
    </button>
  </div>

</main>

<script>

// =========================================================
// BASE DE MESSAGES POUR LE QUIZ
// =========================================================
const messages = [
  { texte: "People like you shouldn't be allowed to marry.", categorie: "Homophobie" },
  { texte: "All Muslims are dangerous and should be watched.", categorie: "Islamophobie" },
  { texte: "Black people are naturally less intelligent.", categorie: "Racisme" },
  { texte: "Women are too emotional to be good leaders.", categorie: "Sexisme" },
  { texte: "Disabled people are a burden on society.", categorie: "Validisme" },
  { texte: "Immigrants are stealing our jobs and our culture.", categorie: "Xénophobie" },
  { texte: "Being gay is a mental illness that can be cured.", categorie: "Homophobie" },
  { texte: "Islam is not a religion, it's a terrorist ideology.", categorie: "Islamophobie" },
  { texte: "Go back to your country, you don't belong here.", categorie: "Xénophobie" },
  { texte: "Women should stay at home and take care of children.", categorie: "Sexisme" },
  { texte: "Wheelchair users should not be in normal schools.", categorie: "Validisme" },
  { texte: "These people bring disease and crime to our country.", categorie: "Xénophobie" },
  { texte: "Two men kissing in public is disgusting.", categorie: "Homophobie" },
  { texte: "All Arabs look like terrorists to me.", categorie: "Islamophobie" },
  { texte: "People with disabilities should not be allowed to vote.", categorie: "Validisme" },
];

const categories = [
  { nom: "Homophobie",    emoji: "🏳️‍🌈" },
  { nom: "Islamophobie",  emoji: "🕌" },
  { nom: "Racisme",       emoji: "✊" },
  { nom: "Sexisme",       emoji: "⚧️" },
  { nom: "Validisme",     emoji: "♿" },
  { nom: "Xénophobie",    emoji: "🌍" },
];

const explications = {
  "Homophobie":   "Ce message cible les personnes LGBTQ+ avec un discours discriminatoire ou haineux.",
  "Islamophobie": "Ce message propage des stéréotypes négatifs ou de la haine envers les musulmans.",
  "Racisme":      "Ce message véhicule des préjugés raciaux et des discriminations fondées sur l'origine.",
  "Sexisme":      "Ce message reproduit des stéréotypes de genre et rabaisse les femmes.",
  "Validisme":    "Ce message discrimine les personnes en situation de handicap.",
  "Xénophobie":   "Ce message exprime de la haine envers les étrangers ou les immigrés.",
};

// =========================================================
// ÉTAT DU QUIZ
// =========================================================
let scoreToi    = 0;
let scoreIA     = 0;
let questionNum = 0;
let totalQ      = 10;
let ordre       = [];
let reponduCette = false;

// =========================================================
// INITIALISATION
// =========================================================
function initQuiz() {
  scoreToi    = 0;
  scoreIA     = 0;
  questionNum = 0;
  reponduCette = false;

  // Mélanger et prendre 10 questions
  ordre = [...messages].sort(() => Math.random() - 0.5).slice(0, totalQ);

  document.getElementById('score-toi').textContent = 0;
  document.getElementById('score-ia').textContent  = 0;
  document.getElementById('q-num').textContent     = 1;
  document.getElementById('quiz-fin').style.display = 'none';
  document.getElementById('quiz-zone').style.display = 'block';

  afficherQuestion();
}

// =========================================================
// AFFICHER UNE QUESTION
// =========================================================
function afficherQuestion() {
  reponduCette = false;
  const q = ordre[questionNum];

  document.getElementById('quiz-message').textContent = '« ' + q.texte + ' »';
  document.getElementById('quiz-feedback').style.display = 'none';
  document.getElementById('q-num').textContent = questionNum + 1;

  // Mélanger les choix
  const choix = [...categories].sort(() => Math.random() - 0.5);
  const container = document.getElementById('quiz-choices');
  container.innerHTML = '';

  choix.forEach(cat => {
    const btn = document.createElement('button');
    btn.className = 'quiz-btn';
    btn.id = 'btn-' + cat.nom;
    btn.innerHTML = cat.emoji + ' ' + cat.nom;
    btn.onclick = () => choisir(cat.nom);
    container.appendChild(btn);
  });
}

// =========================================================
// CHOIX DU JOUEUR
// =========================================================
async function choisir(choix) {
  if (reponduCette) return;
  reponduCette = true;

  const q           = ordre[questionNum];
  const bonne       = q.categorie;
  const joueurJuste = (choix === bonne);

  // Désactiver tous les boutons
  document.querySelectorAll('.quiz-btn').forEach(b => b.disabled = true);

  // Colorer la bonne réponse et la mauvaise
  document.getElementById('btn-' + bonne).classList.add('correct');
  if (!joueurJuste) {
    document.getElementById('btn-' + choix).classList.add('wrong');
  }

  // Demander à l'IA (appel Python via predict.php)
  let iaJuste = false;
  try {
    const fd = new FormData();
    fd.append('texte', q.texte);
    const resp = await fetch('predict.php', { method: 'POST', body: fd });
    const data = await resp.json();
    iaJuste = (data.categorie === bonne);
  } catch(e) {
    // Si l'IA ne répond pas, on considère qu'elle se trompe
    iaJuste = false;
  }

  // Mise à jour des scores
  if (joueurJuste) {
    scoreToi++;
    document.getElementById('score-toi').textContent = scoreToi;
  }
  if (iaJuste) {
    scoreIA++;
    document.getElementById('score-ia').textContent = scoreIA;
  }

  // Feedback
  const titreEl  = document.getElementById('feedback-titre');
  const texteEl  = document.getElementById('feedback-texte');
  const feedback = document.getElementById('quiz-feedback');

  if (joueurJuste) {
    titreEl.textContent = '✅ Bonne réponse ! ';
  } else {
    titreEl.textContent = `❌ C'était : ${bonne}. `;
  }

  texteEl.textContent = explications[bonne] + (iaJuste
    ? " L'IA a aussi trouvé la bonne réponse."
    : " L'IA s'est trompée aussi !");

  feedback.style.display = 'block';

  // Dernière question → cacher "suivant", afficher "fin" au clic
  if (questionNum >= totalQ - 1) {
    document.getElementById('btn-suivant').textContent = '🏆 Voir mon résultat';
    document.getElementById('btn-suivant').onclick = afficherFin;
  }
}

// =========================================================
// QUESTION SUIVANTE
// =========================================================
function questionSuivante() {
  questionNum++;
  if (questionNum >= totalQ) {
    afficherFin();
  } else {
    afficherQuestion();
  }
}

// =========================================================
// FIN DU QUIZ
// =========================================================
function afficherFin() {
  document.getElementById('quiz-zone').style.display = 'none';
  document.getElementById('quiz-fin').style.display  = 'block';
  document.getElementById('score-final').textContent = scoreToi + ' / ' + totalQ;

  let msg = '';
  if (scoreToi === totalQ) {
    msg = '🎉 Parfait ! Tu as tout bon. Tu connais bien ces formes de harcèlement.';
  } else if (scoreToi >= totalQ * 0.7) {
    msg = '👏 Très bon score ! Tu identifies bien la plupart des discours haineux.';
  } else if (scoreToi >= totalQ * 0.4) {
    msg = '🙂 Pas mal ! Consulte la page Prévention pour mieux comprendre chaque catégorie.';
  } else {
    msg = '📚 Ces formes de harcèlement sont parfois subtiles. La page Prévention peut t\'aider !';
  }

  if (scoreToi > scoreIA) {
    msg += ' Et tu as battu l\'IA ! 🤖';
  } else if (scoreToi === scoreIA) {
    msg += ' Égalité avec l\'IA !';
  } else {
    msg += ' L\'IA t\'a devancé cette fois... 🤖';
  }

  document.getElementById('message-final').textContent = msg;
}

// =========================================================
// REJOUER
// =========================================================
function relancerQuiz() {
  initQuiz();
}

// Lancer le quiz au chargement
initQuiz();

</script>
</body>
</html>
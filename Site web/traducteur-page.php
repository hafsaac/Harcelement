<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Traducteur de Bienveillance</title>
  <link rel="stylesheet" href="styles/style.css">
  <style>
    .traducteur-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 20px;
    }

    @media (max-width: 768px) {
      .traducteur-grid { grid-template-columns: 1fr; }
    }

    .traducteur-box {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      padding: 20px;
      min-height: 140px;
      font-size: 1rem;
      line-height: 1.7;
      color: var(--color-text);
    }

    .traducteur-box.haineux {
      border-color: rgba(245,87,108,0.3);
      background: rgba(245,87,108,0.05);
    }

    .traducteur-box.bienveillant {
      border-color: rgba(0,242,254,0.3);
      background: rgba(0,242,254,0.05);
    }

    .box-label {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .box-label.rouge { color: var(--color-pink); }
    .box-label.vert  { color: var(--color-cyan); }

    .fleche-centre {
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2rem;
      color: var(--color-purple);
      padding: 10px 0;
    }

    .exemple-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 50px;
      border: 1px solid var(--color-border);
      background: rgba(255,255,255,0.03);
      color: var(--color-muted);
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.3s ease;
      font-family: 'Inter', sans-serif;
    }

    .exemple-btn:hover {
      background: rgba(102,126,234,0.1);
      border-color: rgba(102,126,234,0.3);
      color: var(--color-text);
    }

    .exemples-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }

    .explication-box {
      background: rgba(102,126,234,0.06);
      border-left: 3px solid var(--color-purple);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      padding: 16px 20px;
      color: var(--color-muted);
      font-size: 0.9rem;
      line-height: 1.7;
      margin-top: 20px;
    }

    .explication-box strong {
      color: var(--color-text);
      display: block;
      margin-bottom: 6px;
    }
  </style>
</head>
<body>

<nav>
  <a href="index.php" class="nav-logo">🛡️ SafeText</a>
  <ul class="nav-links">
    <li><a href="index.php">Analyser</a></li>
    <li><a href="quiz.php">Quiz</a></li>
    <li><a href="visualisations.php">Visualisations</a></li>
    <li><a href="prevention.php">Prévention</a></li>
  </ul>
</nav>

<main>

  <section class="hero">
    <h1>Traducteur de<br>Bienveillance</h1>
    <p>Entrez un message agressif ou haineux — notre outil le transforme
       en une formulation constructive qui exprime le même désaccord
       sans blesser.</p>
  </section>

  <!-- EXEMPLES RAPIDES -->
  <div class="card">
    <div class="card-title">⚡ Exemples rapides</div>
    <div class="exemples-grid">
      <button class="exemple-btn" onclick="chargerExemple(this)">
        Women are not intelligent enough to lead
      </button>
      <button class="exemple-btn" onclick="chargerExemple(this)">
        Immigrants should go back to their country
      </button>
      <button class="exemple-btn" onclick="chargerExemple(this)">
        Gay people are disgusting and should not marry
      </button>
      <button class="exemple-btn" onclick="chargerExemple(this)">
        Disabled people are worthless to society
      </button>
      <button class="exemple-btn" onclick="chargerExemple(this)">
        Muslims are all terrorists
      </button>
    </div>

    <!-- ZONE DE SAISIE -->
    <textarea id="texte-input"
              placeholder="Entrez un message haineux à transformer..."
              maxlength="500"
              style="min-height:100px"></textarea>
    <div class="char-count"><span id="char-count">0</span> / 500</div>

    <button class="btn btn-primary" onclick="traduire()" id="btn-traduire">
      <span id="btn-text">🕊️ Transformer en bienveillance</span>
    </button>
  </div>

  <!-- RÉSULTAT -->
  <div class="card" id="resultat" style="display:none">
    <div class="card-title">✨ Transformation</div>

    <div class="traducteur-grid">
      <div>
        <div class="box-label rouge">⚠️ Message original</div>
        <div class="traducteur-box haineux" id="box-original"></div>
      </div>
      <div>
        <div class="box-label vert">✅ Version bienveillante</div>
        <div class="traducteur-box bienveillant" id="box-traduction"></div>
      </div>
    </div>

    <div class="explication-box">
      <strong>💡 Pourquoi cet outil ?</strong>
      La technologie ne doit pas seulement détecter la haine — elle peut aussi aider à la
      transformer. Ce traducteur montre qu'il est possible d'exprimer un désaccord ou une
      opinion sans recourir à la violence verbale. Derrière chaque message, il y a une
      personne qui reçoit les mots.
    </div>

    <button class="btn btn-primary"
            onclick="document.getElementById('resultat').style.display='none'; document.getElementById('texte-input').value=''; document.getElementById('char-count').textContent='0';"
            style="margin-top:16px; background: rgba(255,255,255,0.06); box-shadow:none; border: 1px solid var(--color-border);">
      🔄 Transformer un autre message
    </button>
  </div>

</main>

<script>
document.getElementById('texte-input').addEventListener('input', function() {
  document.getElementById('char-count').textContent = this.value.length;
});

function chargerExemple(btn) {
  document.getElementById('texte-input').value = btn.textContent.trim();
  document.getElementById('char-count').textContent = btn.textContent.trim().length;
}

async function traduire() {
  const texte = document.getElementById('texte-input').value.trim();
  if (!texte) { alert('Entrez un message à transformer.'); return; }

  const btn = document.getElementById('btn-traduire');
  const btnText = document.getElementById('btn-text');
  btn.disabled = true;
  btnText.innerHTML = '<div class="spinner"></div> Transformation en cours...';

  try {
    const fd = new FormData();
    fd.append('texte', texte);
    const resp = await fetch('traducteur.php', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.erreur) { alert('Erreur : ' + data.erreur); return; }

    document.getElementById('box-original').textContent = data.original;
    document.getElementById('box-traduction').textContent = data.traduction;
    document.getElementById('resultat').style.display = 'block';
    document.getElementById('resultat').scrollIntoView({ behavior: 'smooth' });

  } catch(e) {
    alert('Erreur de connexion.');
  } finally {
    btn.disabled = false;
    btnText.innerHTML = '🕊️ Transformer en bienveillance';
  }
}
</script>
</body>
</html>
<?php
session_start();
require_once 'db.php';

if (!isset($_SESSION['user_id'])) {
    header('Location: auth/connexion.php');
    exit;
}

$user_id = $_SESSION['user_id'];

// Récupérer l'historique
$stmt = $pdo->prepare('SELECT * FROM historique WHERE utilisateur_id = ? ORDER BY date_analyse DESC LIMIT 50');
$stmt->execute([$user_id]);
$historique = $stmt->fetchAll();

$succes = $_SESSION['succes'] ?? '';
unset($_SESSION['succes']);
?>
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Mon compte</title>
  <link rel="stylesheet" href="styles/style.css">
  <style>
    .historique-item {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      padding: 16px 20px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }
    .historique-texte {
      flex: 1;
      font-size: 0.9rem;
      color: var(--color-muted);
      font-style: italic;
      min-width: 200px;
    }
    .historique-badge {
      padding: 4px 12px;
      border-radius: 50px;
      font-size: 0.78rem;
      font-weight: 700;
    }
    .badge-haineux {
      background: rgba(245,87,108,0.15);
      border: 1px solid rgba(245,87,108,0.4);
      color: #f5576c;
    }
    .badge-safe {
      background: rgba(0,242,254,0.1);
      border: 1px solid rgba(0,242,254,0.3);
      color: #00f2fe;
    }
    .historique-date {
      font-size: 0.78rem;
      color: var(--color-muted);
      white-space: nowrap;
    }
    .profil-header {
      display: flex;
      align-items: center;
      gap: 20px;
      margin-bottom: 32px;
    }
    .avatar-cercle {
      width: 70px;
      height: 70px;
      border-radius: 50%;
      background: var(--gradient-main);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.8rem;
      font-weight: 800;
      color: white;
      flex-shrink: 0;
    }
    .empty-state {
      text-align: center;
      padding: 40px;
      color: var(--color-muted);
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
    <li><a href="traducteur-page.php">Traducteur</a></li>
    <li><a href="prevention.php">Prévention</a></li>
    <li><a href="compte.php" class="active">👤 Mon compte</a></li>
  </ul>
</nav>

<main>

  <section class="hero" style="padding:40px 0 20px;">
    <h1>Mon compte</h1>
  </section>

  <?php if ($succes): ?>
    <div style="background:rgba(0,242,254,0.1); border:1px solid rgba(0,242,254,0.3); color:#00f2fe; padding:12px 20px; border-radius:var(--radius-sm); margin-bottom:20px;">
      ✅ <?= htmlspecialchars($succes) ?>
    </div>
  <?php endif; ?>

  <!-- PROFIL -->
  <div class="card">
    <div class="profil-header">
      <div class="avatar-cercle">
        <?= strtoupper(substr($_SESSION['user_nom'], 0, 1)) ?>
      </div>
      <div>
        <div style="font-size:1.4rem; font-weight:800; color:var(--color-text);">
          <?= htmlspecialchars($_SESSION['user_nom']) ?>
        </div>
        <div style="color:var(--color-muted); font-size:0.9rem;">
          <?= htmlspecialchars($_SESSION['user_email']) ?>
        </div>
        <div style="color:var(--color-muted); font-size:0.85rem; margin-top:4px;">
          <?= count($historique) ?> analyse<?= count($historique) > 1 ? 's' : '' ?> effectuée<?= count($historique) > 1 ? 's' : '' ?>
        </div>
      </div>
      <div style="margin-left:auto;">
        <a href="auth/deconnexion.php" style="
          padding:10px 20px;
          border-radius:50px;
          border:1px solid rgba(245,87,108,0.4);
          background:rgba(245,87,108,0.08);
          color:#f5576c;
          text-decoration:none;
          font-size:0.85rem;
          font-weight:600;
        ">🚪 Se déconnecter</a>
      </div>
    </div>
  </div>

  <!-- HISTORIQUE -->
  <div class="card">
    <div class="card-title">📋 Historique de mes analyses</div>

    <?php if (empty($historique)): ?>
      <div class="empty-state">
        <div style="font-size:3rem; margin-bottom:12px;">🔍</div>
        <div>Vous n'avez pas encore effectué d'analyse.</div>
        <a href="index.php" style="color:var(--color-purple); text-decoration:none; font-weight:600;">
          → Analyser un message
        </a>
      </div>
    <?php else: ?>
      <?php foreach ($historique as $h): ?>
        <div class="historique-item">
          <div class="historique-texte">
            « <?= htmlspecialchars(mb_substr($h['texte'], 0, 80)) ?><?= mb_strlen($h['texte']) > 80 ? '…' : '' ?> »
          </div>
          <div class="historique-badge <?= $h['resultat_binaire'] === 'Haineux' ? 'badge-haineux' : 'badge-safe' ?>">
            <?= $h['resultat_binaire'] === 'Haineux' ? '⚠️ Haineux' : '✅ Non haineux' ?>
          </div>
          <?php if ($h['categorie'] && $h['categorie'] !== 'Non applicable'): ?>
            <div style="font-size:0.8rem; color:var(--color-muted);">
              → <?= htmlspecialchars($h['categorie']) ?>
            </div>
          <?php endif; ?>
          <div class="historique-date">
            <?= date('d/m/Y H:i', strtotime($h['date_analyse'])) ?>
          </div>
        </div>
      <?php endforeach; ?>
    <?php endif; ?>
  </div>

</main>

</body>
</html>
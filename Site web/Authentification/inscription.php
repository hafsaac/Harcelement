<?php
session_start();
$erreur = $_SESSION['erreur'] ?? '';
$succes = $_SESSION['succes'] ?? '';
unset($_SESSION['erreur'], $_SESSION['succes']);
?>
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SafeText — Inscription</title>
  <link rel="stylesheet" href="../styles/style.css">
  <style>
    .auth-card {
      max-width: 480px;
      margin: 120px auto 60px;
    }
    .form-group {
      margin-bottom: 18px;
    }
    .form-group label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--color-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .form-group input {
      width: 100%;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      color: var(--color-text);
      font-family: 'Inter', sans-serif;
      font-size: 1rem;
      padding: 12px 16px;
      outline: none;
      transition: border-color 0.3s ease;
      box-sizing: border-box;
    }
    .form-group input:focus {
      border-color: var(--color-purple);
      box-shadow: 0 0 0 3px rgba(102,126,234,0.15);
    }
    .auth-link {
      text-align: center;
      margin-top: 20px;
      font-size: 0.9rem;
      color: var(--color-muted);
    }
    .auth-link a {
      color: var(--color-purple);
      text-decoration: none;
      font-weight: 600;
    }
    .alert {
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      margin-bottom: 20px;
      font-size: 0.9rem;
    }
    .alert-erreur {
      background: rgba(245,87,108,0.1);
      border: 1px solid rgba(245,87,108,0.4);
      color: #f5576c;
    }
    .alert-succes {
      background: rgba(0,242,254,0.1);
      border: 1px solid rgba(0,242,254,0.4);
      color: #00f2fe;
    }
  </style>
</head>
<body>

<nav>
  <a href="../index.php" class="nav-logo">🛡️ SafeText</a>
  <ul class="nav-links">
    <li><a href="../index.php">Analyser</a></li>
    <li><a href="../quiz.php">Quiz</a></li>
    <li><a href="../visualisations.php">Visualisations</a></li>
    <li><a href="../traducteur-page.php">Traducteur</a></li>
    <li><a href="../prevention.php">Prévention</a></li>
    <li><a href="../auth/connexion.php">Connexion</a></li>
  </ul>
</nav>

<div class="auth-card card">
  <div class="card-title">✨ Créer un compte</div>

  <?php if ($erreur): ?>
    <div class="alert alert-erreur">⚠️ <?= htmlspecialchars($erreur) ?></div>
  <?php endif; ?>
  <?php if ($succes): ?>
    <div class="alert alert-succes">✅ <?= htmlspecialchars($succes) ?></div>
  <?php endif; ?>

  <form action="traitement.php" method="POST">
    <input type="hidden" name="action" value="inscription">

    <div class="form-group">
      <label>Prénom</label>
      <input type="text" name="prenom" placeholder="Votre prénom" required>
    </div>
    <div class="form-group">
      <label>Nom</label>
      <input type="text" name="nom" placeholder="Votre nom" required>
    </div>
    <div class="form-group">
      <label>Email</label>
      <input type="email" name="email" placeholder="votre@email.com" required>
    </div>
    <div class="form-group">
      <label>Mot de passe</label>
      <input type="password" name="mot_de_passe" placeholder="Min. 6 caractères" required>
    </div>
    <div class="form-group">
      <label>Confirmer le mot de passe</label>
      <input type="password" name="mot_de_passe2" placeholder="Répétez le mot de passe" required>
    </div>

    <button type="submit" class="btn btn-primary">🚀 Créer mon compte</button>
  </form>

  <div class="auth-link">
    Déjà un compte ? <a href="connexion.php">Se connecter</a>
  </div>
</div>

</body>
</html>
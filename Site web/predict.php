<?php
// =========================================================
// PREDICT.PHP — Reçoit le formulaire et appelle Python
// =========================================================

header('Content-Type: application/json; charset=utf-8');

// Récupération du texte envoyé par le formulaire
$texte = isset($_POST['texte']) ? trim($_POST['texte']) : '';

// Vérification que le texte n'est pas vide
if (empty($texte)) {
    echo json_encode(['erreur' => 'Texte vide']);
    exit;
}

// Sécurisation : on échappe les caractères spéciaux
$texte_secure = escapeshellarg($texte);

// Appel du script Python
$commande = "python3 predict.py $texte_secure 2>&1";
$output   = shell_exec($commande);

if (!$output) {
    echo json_encode(['erreur' => 'Le script Python n\'a pas répondu']);
    exit;
}

// =========================================================
// LECTURE DE LA SORTIE PYTHON
// =========================================================
$resultats = [];
$lignes    = explode("\n", trim($output));

foreach ($lignes as $ligne) {
    if (strpos($ligne, ':') !== false) {
        [$cle, $valeur]    = explode(':', $ligne, 2);
        $resultats[$cle]   = trim($valeur);
    }
}

// Vérification qu'on a bien les résultats attendus
if (!isset($resultats['BINAIRE'])) {
    echo json_encode([
        'erreur' => 'Erreur Python : ' . $output
    ]);
    exit;
}
// Incrémenter le compteur après chaque analyse réussie
$compteur_file = 'compteur.txt';
$compteur = file_exists($compteur_file) ? (int)file_get_contents($compteur_file) : 0;
file_put_contents($compteur_file, $compteur + 1);

// =========================================================
// RÉPONSE JSON vers la page HTML
// =========================================================
echo json_encode([
    'binaire'        => $resultats['BINAIRE'],
    'confiance_bin'  => $resultats['CONFIANCE_BIN'],
    'categorie'      => $resultats['CATEGORIE'],
    'confiance_cat'  => $resultats['CONFIANCE_CAT'],
    'texte'          => $texte
]);
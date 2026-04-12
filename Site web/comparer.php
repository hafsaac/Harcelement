<?php
// =========================================================
// COMPARER.PHP — Compare tous les modèles sur un même texte
// =========================================================

header('Content-Type: application/json; charset=utf-8');

$texte = isset($_POST['texte']) ? trim($_POST['texte']) : '';

if (empty($texte)) {
    echo json_encode(['erreur' => 'Texte vide']);
    exit;
}

$texte_secure = escapeshellarg($texte);
$output = shell_exec("python3 /var/www/html/safetext/comparer.py $texte_secure 2>&1");

if (!$output) {
    echo json_encode(['erreur' => 'Erreur Python']);
    exit;
}

$lignes = explode("\n", trim($output));
$resultats = [];

foreach ($lignes as $ligne) {
    if (strpos($ligne, '|') !== false) {
        $parts = explode('|', $ligne);
        if (count($parts) === 4) {
            $resultats[] = [
                'modele'     => trim($parts[0]),
                'binaire'    => trim($parts[1]),
                'categorie'  => trim($parts[2]),
                'confiance'  => trim($parts[3]),
            ];
        }
    }
}

echo json_encode(['resultats' => $resultats, 'texte' => $texte]);
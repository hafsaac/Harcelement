<?php
// =========================================================
// TRADUCTEUR.PHP — Appelle le script Python de traduction
// =========================================================

header('Content-Type: application/json; charset=utf-8');

$texte = isset($_POST['texte']) ? trim($_POST['texte']) : '';

if (empty($texte)) {
    echo json_encode(['erreur' => 'Texte vide']);
    exit;
}

$texte_secure = escapeshellarg($texte);
$output = shell_exec("python3 /var/www/html/safetext/traducteur.py $texte_secure 2>&1");

if (!$output) {
    echo json_encode(['erreur' => 'Erreur du script Python']);
    exit;
}

echo json_encode([
    'original'    => $texte,
    'traduction'  => trim($output)
]);
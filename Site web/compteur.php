<?php
// =========================================================
// COMPTEUR.PHP — Incrémente le compteur de messages analysés
// Appelé automatiquement après chaque analyse réussie
// =========================================================

$fichier = 'compteur.txt';

// Si le fichier n'existe pas, on le crée avec 0
if (!file_exists($fichier)) {
    file_put_contents($fichier, '0');
}

// On lit la valeur actuelle, on ajoute 1, on sauvegarde
$valeur = (int)file_get_contents($fichier);
$valeur++;
file_put_contents($fichier, $valeur);

// On retourne la nouvelle valeur en JSON
header('Content-Type: application/json');
echo json_encode(['compteur' => $valeur]);
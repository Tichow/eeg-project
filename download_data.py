"""
Téléchargement automatique des fichiers EDF du dataset PhysioNet EEGMMIDB.

Dataset : EEG Motor Movement/Imagery Dataset (EEGMMIDB)
URL     : https://physionet.org/content/eegmmidb/1.0.0/
Format  : EDF+, 64 canaux, 160 Hz, 109 sujets

Runs utilisés :
  R01 → baseline yeux ouverts
  R02 → baseline yeux fermés
  R04, R08, R12 → imagerie motrice (main gauche / droite)
"""

import os
import urllib.request


BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"

# Runs à télécharger
RUNS = [1, 2, 4, 8, 12]

# Sujets à télécharger (S001, S002, S003)
SUBJECTS = [1, 2, 3]


def download_file(url: str, dest_path: str) -> None:
    """Télécharge un fichier depuis une URL vers dest_path, avec barre de progression."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"  [skip] Déjà présent : {dest_path}")
        return

    print(f"  [download] {url}")
    print(f"          → {dest_path}")

    def progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            # Affichage in-place simple
            print(f"\r    {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print()  # Nouvelle ligne après la barre de progression


def download_subject(subject_id: int, runs: list, output_dir: str = "data") -> None:
    """
    Télécharge tous les runs demandés pour un sujet donné.

    Args:
        subject_id : numéro du sujet (ex: 1 → S001)
        runs       : liste des numéros de run (ex: [1, 2, 4])
        output_dir : dossier racine de sortie
    """
    subj_str = f"S{subject_id:03d}"
    subj_dir = os.path.join(output_dir, subj_str)

    print(f"\n=== Sujet {subj_str} ===")

    for run_id in runs:
        filename = f"{subj_str}R{run_id:02d}.edf"
        url = f"{BASE_URL}{subj_str}/{filename}"
        dest = os.path.join(subj_dir, filename)
        download_file(url, dest)


def main():
    print("=== Téléchargement du dataset PhysioNet EEGMMIDB ===")
    print(f"Sujets : {[f'S{s:03d}' for s in SUBJECTS]}")
    print(f"Runs   : {[f'R{r:02d}' for r in RUNS]}")
    print()

    for subject_id in SUBJECTS:
        download_subject(subject_id, RUNS)

    print("\nTéléchargement terminé. Fichiers dans : data/")


if __name__ == "__main__":
    main()

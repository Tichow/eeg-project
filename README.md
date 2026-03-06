# EEG Analysis — PhysioNet EEGMMIDB

Exploration du dataset EEG Motor Movement/Imagery Database de PhysioNet.
Interface graphique PyQt5, analyse offline uniquement.

## Prérequis

- Python 3.11+
- Un environnement virtuel (voir ci-dessous)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
source .venv/bin/activate
python main.py
```

## Fonctionnalités

### Télécharger les données PhysioNet

Depuis la homepage, cliquer sur **"Télécharger les données"**.

| Champ | Description |
|---|---|
| Sujets (de / à) | Plage de sujets à télécharger (S001–S109) |
| Runs | Sélection multiple des runs (R01–R14) |
| Dossier de destination | Chemin local (défaut : `data/`) |

Les fichiers EDF sont téléchargés depuis PhysioNet via MNE-Python et stockés dans :
```
data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{NNN}/
```

La barre de progression avance fichier par fichier. Les fichiers déjà présents localement sont skippés instantanément.

### Explorer les données

Depuis la homepage, cliquer sur **"Explorer les données"**.

- La liste des sujets disponibles localement s'affiche automatiquement
- Sélectionner un sujet pour voir ses fichiers EDF dans un tableau
- Les métadonnées EDF (durée, fréquence d'échantillonnage, nombre de canaux, annotations) se chargent progressivement sans bloquer l'interface
- Bouton **"Actualiser"** pour re-scanner après un nouveau téléchargement
- **"Parcourir…"** pour pointer vers un autre dossier de données

| Colonne | Description |
|---|---|
| Run | Identifiant du run (R01–R14) |
| Description | Type de tâche correspondant au run |
| Durée | Durée de l'enregistrement en secondes |
| Fréquence | Fréquence d'échantillonnage (Hz) |
| Canaux | Nombre de canaux EEG |
| Annotations | Nombre de marqueurs d'événements (T0/T1/T2) |
| Taille | Taille du fichier sur disque |

### Visualiser un signal EEG

Depuis l'explorateur, **double-cliquer sur un run** pour ouvrir la vue signal.

- Les 10 premiers canaux sont affichés par défaut avec un offset vertical (un canal par ligne)
- Les noms des canaux sont indiqués sur l'axe Y
- Les annotations sont marquées par des lignes verticales colorées : T0 (gris), T1 (bleu), T2 (rouge)
- **Sélecteur de canaux** (panneau gauche) : cocher/décocher pour afficher/masquer des canaux
- **Toolbar Matplotlib** : zoom, pan, sauvegarde de l'image
- Retour au browser → re-double-clic sur le même fichier = rechargement instantané (cache)

## Dataset

**PhysioNet EEGMMIDB** — [https://physionet.org/content/eegmmidb/1.0.0/](https://physionet.org/content/eegmmidb/1.0.0/)

- 109 sujets (S001–S109)
- 14 runs par sujet, format EDF+
- 64 canaux EEG (système 10-20), 160 Hz
- Annotations : T0 (repos), T1/T2 (tâches motrices selon le run)

| Run | Description |
|---|---|
| R01 | Baseline — yeux ouverts |
| R02 | Baseline — yeux fermés |
| R03, R07, R11 | Imagerie motrice — main G/D (cible) |
| R04, R08, R12 | Imagerie motrice — main G/D (alternance) |
| R05, R09, R13 | Imagerie motrice — 2 mains / 2 pieds (cible) |
| R06, R10, R14 | Imagerie motrice — 2 mains / 2 pieds (alternance) |

Licence : [Open Data Commons Attribution License (ODC-By)](https://physionet.org/content/eegmmidb/1.0.0/)

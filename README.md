# Projet EEG — Visualisation de l'effet Berger

Reproduction des analyses de l'article *"EEG 101 using OpenBCI Ultracortex"* (Towards Data Science)
sur des données publiques, avant application sur nos signaux OpenBCI Cyton.

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)**
- 109 sujets, 64 canaux, 160 Hz, format EDF+
- Runs utilisés :
  - `R01` — baseline yeux ouverts
  - `R02` — baseline yeux fermés
  - `R04 / R08 / R12` — imagerie motrice (main gauche / droite)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

**1. Télécharger les données**

```bash
python download_data.py
```

Télécharge S001–S003 depuis PhysioNet dans le dossier `data/` (non commité).

**2. Lancer l'analyse**

```bash
python eeg_analysis.py
```

Produit 4 figures :
1. Signal brut (time series)
2. PSD avec bandes δ / θ / α / β
3. Comparaison yeux ouverts vs fermés → pic alpha ~10 Hz (effet Berger)
4. Rythme mu sur C3/C4 (imagerie motrice)

## Matériel cible

OpenBCI Cyton — 8 canaux, 250 Hz, électrodes sèches ThinkPulse.
Les fonctions dans `eeg_analysis.py` acceptent `sfreq` et `channels` en paramètres
pour s'adapter facilement à nos données.

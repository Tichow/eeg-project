# EEG Analysis — PhysioNet EEGMMIDB & OpenBCI Cyton

Interface graphique complète pour l'analyse EEG : exploration du dataset PhysioNet EEGMMIDB, acquisition temps réel via OpenBCI Cyton, classification par imagerie motrice (CSP+LDA) et prédiction BCI en direct.

Projet ISMIN – École des Mines de Saint-Étienne, 2025–2026.
Auteurs : Fabien Saliba, Mattéo Quintaneiro, Némo Kardassevitch.

## Prérequis

- Python 3.14+
- Un environnement virtuel (voir ci-dessous)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances principales : PyQt5, PyQtGraph, MNE-Python, NumPy, SciPy, scikit-learn, BrainFlow, pyserial, edfio.

## Lancement

```bash
source .venv/bin/activate
python main.py
```

## Fonctionnalités

La homepage présente cinq cartes de navigation :

| Carte | Description |
|---|---|
| Télécharger les données | Téléchargement du dataset PhysioNet |
| Explorer les données | Explorateur EDF avec métadonnées |
| Visualiser un signal | Analyse et prétraitement offline |
| Acquisition EEG | Enregistrement live avec OpenBCI Cyton |
| Préréglages d'électrodes | Gestion des montages 8 canaux |

---

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

---

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

---

### Visualiser un signal EEG

Depuis l'explorateur, **double-cliquer sur un run** pour ouvrir la vue signal.

#### Affichage

- Les 10 premiers canaux sont affichés par défaut avec un offset vertical (un canal par ligne)
- Les noms des canaux sont indiqués sur l'axe Y
- Les annotations sont marquées par des lignes verticales colorées : T0 (gris), T1 (bleu), T2 (rouge)
- **Sélecteur de canaux** (panneau gauche) : cocher/décocher pour afficher/masquer des canaux
- **Interaction native PyQtGraph** : scroll pour zoomer, drag pour panner, clic droit pour reset/export

#### Prétraitement

| Paramètre | Description |
|---|---|
| Bandpass | Filtre passe-bande (Hz bas / Hz haut) |
| Notch | Filtre coupe-bande secteur (ex. 50 Hz) |
| Reréférencement | Mode `average` ou nom de canal de référence |
| Signal traité | Bascule entre signal brut et signal prétraité |

#### Outils d'analyse (panneau droit)

| Outil | Description |
|---|---|
| **ICA** | Analyse en composantes indépendantes — décomposition et exclusion de composantes artefactuelles |
| **Analyse fréquentielle** | PSD par canal, détection de l'effet Berger (ratio alpha yeux-fermés/yeux-ouverts) |
| **ERP / ERD-ERS** | Moyennage évoqué par événements et cartes temps-fréquence (imagerie motrice) |
| **Topomaps** | Cartes de distribution spatiale de la puissance par bande fréquentielle |
| **Détection d'artefacts** | Rejet automatique d'epochs par seuil d'amplitude |
| **Export EDF** | Sauvegarde du signal prétraité au format EDF |

---

### Acquisition EEG — OpenBCI Cyton

Depuis la homepage, cliquer sur **"Acquisition EEG"**.

Interface d'enregistrement temps réel via un casque **OpenBCI Cyton** (8 canaux, 250 Hz) connecté en USB/série. Utilise [BrainFlow](https://brainflow.org) pour le streaming.

#### Connexion et configuration

- Sélectionner le **port série** du Cyton (`/dev/tty…` sur macOS/Linux)
- Configurer le **gain** ADS1299 (1, 2, 4, 6, 8, 12, 24 — défaut 24)
- Assigner les **noms d'électrodes** (système 10-20) à chaque canal physique
- Indicateurs de qualité de signal par canal : vert (ok), orange (rail), rouge (plat)

#### Enregistrement

- **Protocole** : choisir un protocole d'acquisition (imagerie motrice, effet Berger…)
- **Fenêtre de cues** : affichage plein écran des consignes au sujet pendant la session
- **Sujet** : numéro de sujet pour nommer automatiquement le fichier de sortie
- Cliquer **"Démarrer l'enregistrement"** — le fichier EDF est sauvegardé à l'arrêt

Les enregistrements sont stockés dans `data/custom/` au format EDF, compatibles avec la vue signal et les scripts de classification.

---

### Préréglages d'électrodes

Depuis la homepage, cliquer sur **"Préréglages d'électrodes"**.

Outil de gestion des montages 8 canaux pour le Cyton. Visualisation interactive sur une tête 2D (vue du dessus, système 10-20).

- **Créer / renommer / supprimer** des presets nommés
- Assigner une électrode standard (Fp1…O2) à chacun des 8 pins Cyton
- La carte de tête se met à jour en temps réel
- Les presets sont persistants (sauvegardés localement) et disponibles dans la vue Acquisition

---

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

---

## Classification Motor Imagery (CSP + LDA)

Pipeline de classification offline pour l'imagerie motrice EEG, utilisant **Common Spatial Patterns (CSP)** couplé à une **Linear Discriminant Analysis (LDA)**. Aucun GPU requis — tourne en secondes sur un laptop.

### Setup d'electrodes (8 canaux, Cyton)

Le preset `MotorImagery` place les 8 canaux autour du cortex moteur :

```
        FC1    FC2       ← premoteur
     C3    Cz    C4      ← moteur primaire
        CP1    CP2       ← somatosensoriel
           Pz            ← parietal
```

| Pin Cyton | Electrode | Role |
|-----------|-----------|------|
| CH1 | C3 | Moteur primaire gauche |
| CH2 | FC1 | Premoteur gauche |
| CH3 | C4 | Moteur primaire droit |
| CH4 | CP1 | Somatosensoriel gauche |
| CH5 | Cz | Moteur midline (pieds) |
| CH6 | CP2 | Somatosensoriel droit |
| CH7 | FC2 | Premoteur droit |
| CH8 | Pz | Parietal midline |

### Pipeline

```
EDF → Bandpass 8-30 Hz → Epoch [0.5s, 3.5s] → Rejet artefacts (500 µV) → CSP (6 comp.) → LDA → 5-fold CV
```

Les 8 memes canaux sont extraits des 64 canaux PhysioNet pour que les resultats soient directement transferables aux donnees Cyton.

### Scripts

#### Classifier sur PhysioNet (validation)

```bash
# Un seul sujet
python -m src.scripts.classify_physionet --subjects 1 --task left_vs_right

# Plage de sujets
python -m src.scripts.classify_physionet --subjects 1-10 --task left_vs_right

# Hands vs Feet
python -m src.scripts.classify_physionet --subjects 1-10 --task hands_vs_feet

# Avec sauvegarde du modele
python -m src.scripts.classify_physionet --subjects 1 --task left_vs_right --save-model

# Utiliser les 64 canaux PhysioNet au lieu de 8
python -m src.scripts.classify_physionet --subjects 1 --task left_vs_right --channels 64
```

#### Classifier ses propres donnees (Cyton)

```bash
# Cross-validation sur ses propres enregistrements
python -m src.scripts.classify_custom --subject MATTEO2 --runs 3 --task left_vs_right

# Transfert : appliquer un modele PhysioNet sur ses donnees
python -m src.scripts.classify_custom --subject MATTEO2 --runs 3 --transfer models/S001_left_vs_right_csp_lda.pkl
```

#### Options

| Option | Description |
|--------|-------------|
| `--subjects` | Sujet(s) PhysioNet : `1`, `1-10`, ou `1,3,5` |
| `--task` | `left_vs_right` (imagerie main G/D) ou `hands_vs_feet` (2 mains / 2 pieds) |
| `--channels` | `8` (subset moteur, defaut) ou `64` (tous les canaux PhysioNet) |
| `--save-model` | Sauvegarder le modele entraine dans `models/` |
| `--data-path` | Chemin vers les donnees (defaut : `data/`) |
| `--transfer` | Chemin vers un modele `.pkl` pre-entraine (mode transfert) |

### Resultats benchmark (103 sujets PhysioNet, 8 canaux)

Benchmark complet sur les sujets de PhysioNet en utilisant uniquement les 8 canaux du setup Motor Imagery (C3, FC1, C4, CP1, Cz, CP2, FC2, Pz). Cross-validation stratifiee 5-fold, within-subject. S109 exclu (donnees corrompues), 5 sujets supplementaires exclus pour donnees manquantes.

#### Left vs Right (imagerie main gauche / droite)

| Metrique | Valeur |
|----------|--------|
| Sujets testes | 108 (S109 exclu — donnees corrompues) |
| **Accuracy moyenne** | **63.0% +/- 16.7%** |
| Mediane | 62.2% |
| Min / Max | 31.1% / 97.8% |
| Chance level | 50% |

Distribution :

```
  < 50% :  24 sujets (22.2%)  ███████████
 50-59% :  22 sujets (20.4%)  ██████████
 60-69% :  29 sujets (26.9%)  █████████████
 70-79% :  16 sujets (14.8%)  ███████
 80-89% :   6 sujets ( 5.6%)  ██
 90%+   :  11 sujets (10.2%)  █████
```

Top 5 : S029 (97.8%), S062 (97.8%), S094 (95.6%), S002 (93.3%), S007 (93.3%)

#### Hands vs Feet (imagerie 2 mains / 2 pieds)

| Metrique | Valeur |
|----------|--------|
| Sujets testes | 108 (S109 exclu) |
| **Accuracy moyenne** | **71.1% +/- 16.0%** |
| Mediane | 70.0% |
| Min / Max | 35.6% / 100.0% |
| Chance level | 50% |

Distribution :

```
  < 50% :  10 sujets ( 9.3%)  ████
 50-59% :  18 sujets (16.7%)  ████████
 60-69% :  26 sujets (24.1%)  ████████████
 70-79% :  17 sujets (15.7%)  ███████
 80-89% :  20 sujets (18.5%)  █████████
 90%+   :  17 sujets (15.7%)  ███████
```

Top 5 : S001 (100%), S035 (97.8%), S042 (97.8%), S062 (97.8%), S072 (97.8%)

#### Donnees custom (OpenBCI Cyton)

Le meilleur sujet custom (MATTEO5, 3 runs combinés, hands vs feet) a atteint **93.3% ± 5.4%** en cross-validation (89/90 epochs retenus). Ce résultat place MATTEO5 dans le top 10% des sujets PhysioNet sur la même tâche.

#### Analyse

- **Hands vs Feet est plus facile** (+8 points de moyenne) car les zones cerebrales impliquees (zone main = C3/C4 lateral, zone pied = Cz midline) sont spatialement plus distinctes.
- **Grande variabilite inter-sujets** : certains sujets atteignent >95% avec seulement 8 canaux, d'autres restent au niveau du hasard. C'est typique en BCI — la capacite a produire des patterns moteurs discriminables varie fortement entre individus (phenomene de "BCI illiteracy").
- **8 canaux suffisent** pour obtenir des resultats comparables a la litterature. Le setup dense autour du cortex moteur (FC1/FC2, C3/Cz/C4, CP1/CP2, Pz) capture l'essentiel de l'information ERD/ERS en mu (8-12 Hz) et beta (13-30 Hz).
- **Generalisation inter-sujets limitee** : sans recalibration, la precision chute autour de 55% sur un nouveau sujet. Chaque utilisateur nécessite une session de calibration dédiée.

### Architecture classification

```
src/
  models/
    classification_data.py           # ClassificationConfig, ClassificationResult
  services/
    eeg_classification_service.py    # CSP+LDA pipeline (load, preprocess, epoch, CV, save)
  scripts/
    classify_physionet.py            # Benchmark PhysioNet (within-subject)
    classify_custom.py               # Classification donnees custom + mode transfert
```

Le service reutilise les services existants (`EEGPreprocessService`, `EEGEpochService`, `EEGArtifactService`) — pas de code duplique. Les modeles entraines sont sauvegardes en pickle dans `models/`.

---

## Prediction BCI temps reel

Interface de controle par la pensee en temps reel. Charge un modele CSP+LDA entraine, connecte le casque Cyton, et predit en continu la commande motrice du sujet avec un retour visuel interactif.

### Utilisation

1. Depuis l'accueil, cliquer sur **"Prediction BCI"**
2. **Connexion** : selectionner le port serie du Cyton et connecter
3. **Modele** : charger un fichier `.pkl` (ex: `models/MATTEO5_hands_vs_feet_csp_lda.pkl`)
4. **Commandes** : nommer les deux classes (ex: "Mains" / "Pieds")
5. Cliquer **"Demarrer la prediction"**
6. Le buffer se remplit pendant 3 secondes, puis les predictions commencent (~2 par seconde)

### Interface

- **Mini-jeu** : une balle se deplace horizontalement en fonction de la probabilite predite. Les zones cibles s'allument quand la confiance depasse le seuil
- **Barres de confiance** : probabilites T1 et T2 en temps reel
- **Statistiques** : nombre de predictions, predictions confiantes, derniere prediction

### Reglages

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| Lissage (EMA) | 0.3 | Coefficient de moyenne mobile exponentielle. Plus bas = plus lisse, plus haut = plus reactif |
| Seuil confiance | 0.60 | Probabilite minimum pour activer une commande (zone s'allume, balle atteint le bord) |

### Fonctionnement technique

Le systeme accumule 3 secondes de donnees EEG (750 echantillons a 250 Hz) dans un buffer circulaire. Toutes les 0.5 secondes, le buffer est filtre (bandpass causal 8-30 Hz + notch 50 Hz) et passe au pipeline CSP+LDA pour obtenir une prediction et des probabilites. Le lissage EMA reduit le bruit entre predictions consecutives.

### Architecture

```
src/
  workers/
    prediction_stream_worker.py    # QThread: buffer circulaire + filtrage causal + prediction
  views/
    prediction_view.py             # Vue BCI: connexion, modele, mini-jeu, barres de confiance
```

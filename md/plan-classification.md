# Plan de classification MI-EEG — PhysioNet → OpenBCI

Pipeline complet pour entraîner un modèle EEGNet sur les données PhysioNet (64 canaux) avec channel dropout, puis le tester/fine-tuner sur nos données OpenBCI (8 canaux).

---

## Vue d'ensemble du pipeline

```
PhysioNet EEGMMIDB (50 sujets, 64 canaux, 160 Hz)
    ↓
Prétraitement (bandpass 4–40 Hz, notch 60 Hz, CAR)
    ↓
Extraction des époques MI (0.5–4.0 s après onset T1/T2)
    ↓
Rejet d'artefacts (seuil 100 µV crête-à-crête)
    ↓
Normalisation Z-score par canal par époque
    ↓
Euclidean Alignment par sujet
    ↓
EEGNet (64 canaux, channel dropout pendant l'entraînement)
    ↓
Modèle pré-entraîné sauvegardé
    ↓
Test sur OpenBCI (8 canaux → masquage des 56 absents)
    ↓
Fine-tuning spécifique au sujet
```

---

## Étape 0 — Dépendances

Ajouter à `requirements.txt` :

```
torch>=2.0
scikit-learn>=1.3
```

PyTorch ARM natif tourne sur MacBook M4 en CPU pour le prototypage.
Pour l'entraînement complet (50 sujets, 300 époques) : **utiliser Google Colab avec GPU** (gratuit, T4).

### Setup Colab

```python
# En haut du notebook Colab :
!pip install mne torch scikit-learn scipy numpy
# Uploader le dossier src/ ou cloner le repo
!git clone https://github.com/Tichow/eeg-project.git
```

---

## Étape 1 — Téléchargement des données PhysioNet

**Fichier** : `src/services/eeg_download_service.py` (existant)

**Action** : Télécharger les runs MI pour 50 sujets.

```python
# Sujets à exclure (anomalies documentées)
EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104, 106}

# Runs d'imagerie motrice
MI_RUNS_HANDS = [4, 8, 12]   # MI mains : T1=gauche, T2=droite
MI_RUNS_FEET  = [6, 10, 14]  # MI pieds : T1=deux poings, T2=deux pieds
MI_RUNS = MI_RUNS_HANDS + MI_RUNS_FEET

# Sujets valides (50 premiers)
VALID_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS][:50]
```

**Réutilise** : `EEGDownloadService.download_subject(subject, runs, path)`

**Sortie** : ~300 fichiers EDF dans `data/`

---

## Étape 2 — Prétraitement et extraction des époques

**Fichier à créer** : `src/services/eeg_dataset_service.py`

Ce service orchestre le pipeline existant pour produire un dataset prêt à classifier.

### 2.1 — Chargement + prétraitement

Réutilise les services existants :

```python
signal = EEGSignalService.load_signal(edf_path)         # → SignalData (64ch, Volts)
signal = EEGPreprocessService.apply(signal, config)       # → SignalData filtré
```

**Config de prétraitement** :

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Bandpass | 4–40 Hz | Supprime dérives + EMG, garde mu (8-13) et beta (13-30) |
| Notch | 60 Hz pour PhysioNet, 50 Hz pour OpenBCI | Secteur USA vs France |
| Re-référençage | CAR (average) | Aligné avec la littérature |
| Ordre filtre | 4 (Butterworth) | Déjà implémenté dans `EEGPreprocessService` |

### 2.2 — Extraction des époques

Réutilise `EEGEpochService.extract()` :

```python
epochs = EEGEpochService.extract(signal, tmin=0.0, tmax=4.5)
# → EpochData (n_epochs, 64, 720) à 160 Hz
```

**Attention au tmin/tmax** : le service coupe autour de chaque annotation. Les annotations PhysioNet incluent T0 (repos), T1 et T2 (tâches). Il faut :
- Garder les époques avec `tmin=0.5, tmax=4.0` après l'onset T1/T2 (3.5s utiles, 560 samples)
- Le service actuel coupe autour de TOUTES les annotations → il faut **filtrer par label après** pour ne garder que T1 et T2

```python
# Après extraction
task_mask = [l in ("T1", "T2") for l in epochs.labels]
X = epochs.data[task_mask]     # (n_trials, 64, 560)
y = [l for l, m in zip(epochs.labels, task_mask) if m]  # ["T1", "T2", ...]
```

### 2.3 — Rejet d'artefacts

Réutilise `EEGArtifactService` :

```python
bad = EEGArtifactService.detect_by_threshold(epochs, threshold_uv=100)
epochs = EEGArtifactService.apply_threshold_rejection(epochs, bad)
```

### 2.4 — Gestion des labels par type de run

**Point critique** : T1/T2 n'ont pas la même signification selon le run.

| Runs | T1 | T2 |
|------|----|----|
| 4, 8, 12 (mains) | Main gauche | Main droite |
| 6, 10, 14 (pieds) | Deux poings | Deux pieds |

Le service doit **remapper les labels** :

```python
if run in MI_RUNS_HANDS:
    label_map = {"T1": "left_hand", "T2": "right_hand"}
elif run in MI_RUNS_FEET:
    label_map = {"T1": "both_fists", "T2": "both_feet"}
```

Pour l'entraînement multi-tâche, on aura 4 classes : `left_hand`, `right_hand`, `both_fists`, `both_feet`.
Pour la classification binaire main gauche/droite, on filtre sur `left_hand` et `right_hand` seulement.

### 2.5 — Normalisation

```python
# Z-score par canal par époque
for i in range(X.shape[0]):
    for ch in range(X.shape[1]):
        X[i, ch] = (X[i, ch] - X[i, ch].mean()) / (X[i, ch].std() + 1e-8)
```

### 2.6 — Sortie de l'étape 2

```python
# Par sujet :
X: np.ndarray  # (n_trials, 64, 560) — float32, normalisé
y: np.ndarray  # (n_trials,) — entiers 0/1 (binaire) ou 0/1/2/3 (multi-classe)
```

**Total estimé pour 50 sujets** :
- Runs mains : ~45 essais × 50 sujets = ~2250 essais (2 classes)
- Runs pieds : ~45 essais × 50 sujets = ~2250 essais (2 classes)
- Total : ~4500 essais (4 classes)

---

## Étape 3 — Euclidean Alignment

**Fichier à créer** : `src/services/eeg_alignment_service.py`

L'EA réduit la variabilité inter-sujets en alignant les distributions de covariance.

### Algorithme

Pour chaque sujet s :

```python
import numpy as np
from scipy.linalg import sqrtm, inv

def euclidean_alignment(X_subject):
    """
    X_subject: (n_trials, n_channels, n_times)
    Retourne X aligné de même shape.
    """
    n_trials, n_ch, n_t = X_subject.shape

    # 1. Calculer la matrice de covariance moyenne
    R_mean = np.zeros((n_ch, n_ch))
    for i in range(n_trials):
        R_mean += X_subject[i] @ X_subject[i].T / n_t
    R_mean /= n_trials

    # 2. Calculer R_mean^(-1/2)
    R_inv_sqrt = inv(sqrtm(R_mean)).real

    # 3. Appliquer le blanchiment
    X_aligned = np.zeros_like(X_subject)
    for i in range(n_trials):
        X_aligned[i] = R_inv_sqrt @ X_subject[i]

    return X_aligned, R_inv_sqrt  # Sauvegarder R_inv_sqrt pour le test
```

**Important** : sauvegarder `R_inv_sqrt` par sujet pour pouvoir l'appliquer aux données OpenBCI au moment du fine-tuning.

---

## Étape 4 — Architecture EEGNet (PyTorch)

**Fichier à créer** : `src/models/eegnet.py`

### Architecture EEGNet-8,2

```
Input: (batch, 1, 64, 560)     ← (batch, 1, n_channels, n_times)
    ↓
Conv2d temporel (1→F1, (1, kernel_length))     F1=8, kernel_length=80 (0.5s à 160Hz)
BatchNorm2d
    ↓
DepthwiseConv2d (F1→F1*D, (n_channels, 1))    D=2, groups=F1
BatchNorm2d → ELU → AvgPool2d(1, 4) → Dropout(0.5)
    ↓
SeparableConv2d (F1*D→F2, (1, 16))            F2=16
BatchNorm2d → ELU → AvgPool2d(1, 8) → Dropout(0.5)
    ↓
Flatten → Linear(F2 * remaining_time, n_classes)
```

### Channel Dropout (clé pour la flexibilité)

Pendant l'entraînement, on met aléatoirement des canaux entiers à zéro :

```python
class ChannelDropout(nn.Module):
    def __init__(self, p=0.3):
        """p = probabilité de masquer chaque canal."""
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (batch, 1, n_channels, n_times)
        if not self.training:
            return x
        mask = torch.bernoulli(torch.full((x.size(0), 1, x.size(2), 1), 1 - self.p))
        return x * mask.to(x.device)
```

Appliqué **avant** le premier Conv2d. Ainsi le modèle apprend à fonctionner avec des canaux manquants → au test, on peut masquer les 56 canaux absents de l'OpenBCI.

### Hyperparamètres

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| F1 | 8 | Filtres temporels |
| D | 2 | Multiplicateur depthwise |
| F2 | 16 | Filtres séparables |
| kernel_length | 80 | 0.5s à 160 Hz |
| dropout_rate | 0.5 | Régularisation standard |
| channel_dropout | 0.3 | 30% des canaux masqués aléatoirement |
| n_classes | 2 ou 4 | Binaire (main G/D) ou multi-classe |

---

## Étape 5 — Entraînement

**Fichier à créer** : `src/scripts/train_eegnet.py`

### 5.1 — Split des données

Cross-validation **par sujet** (leave-subjects-out) :

```
50 sujets → 5 folds de 10 sujets
Fold 1 : Train sur sujets 1-40, Val sur sujets 41-50
Fold 2 : Train sur sujets 1-10+21-50, Val sur sujets 11-20
...
```

### 5.2 — Paramètres d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Optimiseur | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Époques max | 300 |
| Early stopping | patience=50 sur val_loss |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=20) |

### 5.3 — Environnement d'exécution

| Environnement | Usage | Temps estimé |
|---------------|-------|--------------|
| MacBook M4 (CPU) | Prototypage sur 5 sujets | ~10 min |
| Google Colab (T4 GPU) | Entraînement complet 50 sujets | ~30-60 min |

### 5.4 — Métriques à tracker

- **Accuracy** par fold et moyenne
- **Confusion matrix** (normalisée)
- **Cohen's Kappa** (standard BCI Competition)
- **Loss curves** (train vs val)

### 5.5 — Sauvegarde du modèle

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'channel_names': ch_names,  # Liste des 64 canaux dans l'ordre
    'n_classes': n_classes,
    'sfreq': 160,
    'preprocessing': {
        'bandpass': (4, 40),
        'notch': 60,
        'reref': 'average',
        'epoch_tmin': 0.5,
        'epoch_tmax': 4.0,
    }
}, 'models/eegnet_physionet.pt')
```

### Performances attendues

| Configuration | Accuracy attendue | Source |
|---------------|-------------------|--------|
| 64 canaux, cross-sujet, 2 classes | 70–82% | Wang et al., 2020 |
| 64 canaux, cross-sujet, 4 classes | 55–65% | Estimation |
| Avec channel dropout → test 8 canaux | 60–72% | Estimation (dégradation ~10%) |

---

## Étape 6 — Test sur données OpenBCI

**Fichier à créer** : `src/scripts/test_on_openbci.py`

### 6.1 — Mapping des canaux

```python
# Canaux OpenBCI → indices dans les 64 canaux PhysioNet
OPENBCI_TO_PHYSIONET = {
    'C3':  8,   # index dans les 64 canaux EDF
    'Cz':  10,
    'C4':  12,
    'FC3': 1,
    'FC4': 5,
    'CP3': 15,
    'CP4': 19,
    'FCz': 3,
}
```

### 6.2 — Préparation des données OpenBCI

```python
# 1. Charger le recording OpenBCI (.npy + .json)
# 2. Rééchantillonner 250 Hz → 160 Hz
# 3. Appliquer le MÊME prétraitement (bandpass 4-40, notch 50 Hz, CAR)
# 4. Extraire les époques (même timing : 0.5–4.0s après onset)
# 5. Normaliser Z-score
# 6. Construire le tenseur 64 canaux avec zéros sauf aux 8 positions OpenBCI
# 7. Appliquer Euclidean Alignment

X_64 = np.zeros((n_trials, 64, 560))
for ch_name, idx in OPENBCI_TO_PHYSIONET.items():
    openbci_ch_idx = openbci_channels.index(ch_name)
    X_64[:, idx, :] = X_openbci[:, openbci_ch_idx, :]
```

### 6.3 — Inférence

```python
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_64).unsqueeze(1)  # (batch, 1, 64, 560)
    logits = model(X_tensor)
    predictions = logits.argmax(dim=1)
```

---

## Étape 7 — Fine-tuning sur données OpenBCI

**Fichier à créer** : `src/scripts/finetune_eegnet.py`

### 7.1 — Stratégie de gel

```python
# Geler les deux premiers blocs (conv temporelle + depthwise)
for name, param in model.named_parameters():
    if 'block1' in name or 'block2' in name:
        param.requires_grad = False

# Seuls le bloc séparable et la couche FC sont entraînables
```

### 7.2 — Paramètres de fine-tuning

| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 1e-4 (10x plus faible) |
| Époques | 50–100 |
| Batch size | 16 (peu de données) |
| Early stopping | patience=20 |
| Données nécessaires | ~40 essais par classe minimum |
| Validation | 4-fold cross-validation intra-sujet |

### 7.3 — Gain attendu

+5 à 15% d'accuracy par rapport au modèle cross-sujet appliqué directement (Wang et al., 2020).

---

## Résumé des fichiers à créer

```
src/
├── models/
│   └── eegnet.py                      # Architecture EEGNet + ChannelDropout
├── services/
│   ├── eeg_dataset_service.py         # Orchestration prétraitement → dataset
│   └── eeg_alignment_service.py       # Euclidean Alignment
└── scripts/
    ├── prepare_dataset.py             # Télécharge + prépare les données
    ├── train_eegnet.py                # Entraînement cross-sujet
    ├── test_on_openbci.py             # Test sur nos données
    └── finetune_eegnet.py             # Fine-tuning spécifique au sujet

models/                                # Dossier pour les poids sauvegardés
    └── eegnet_physionet.pt
```

---

## Checklist de progression

- [ ] Étape 0 : Installer torch + scikit-learn
- [ ] Étape 1 : Télécharger 50 sujets PhysioNet (runs 4,6,8,10,12,14)
- [ ] Étape 2 : Créer `eeg_dataset_service.py` — valider sur 3 sujets
- [ ] Étape 3 : Implémenter Euclidean Alignment
- [ ] Étape 4 : Implémenter EEGNet avec channel dropout
- [ ] Étape 5a : Prototyper l'entraînement sur 5 sujets (MacBook M4, CPU)
- [ ] Étape 5b : Entraîner sur 50 sujets (Colab GPU)
- [ ] Étape 5c : Évaluer les performances (accuracy, kappa, confusion matrix)
- [ ] Étape 6 : Tester sur données OpenBCI existantes
- [ ] Étape 7 : Fine-tuning avec nouvelles acquisitions OpenBCI

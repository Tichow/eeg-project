# Protocole PhysioNet EEGMMIDB — Reproduction exacte

Guide complet pour reproduire le protocole expérimental du dataset **EEG Motor Movement/Imagery** (PhysioNet EEGMMIDB) avec un casque OpenBCI. Ce document décrit exactement ce qui a été fait dans l'étude originale, puis comment l'adapter à notre matériel.

**Source :** Schalk et al., "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System", IEEE Trans. Biomedical Engineering, 2004. Dataset : https://physionet.org/content/eegmmidb/1.0.0/

---

## 1. Vue d'ensemble du dataset original

| Paramètre | Valeur |
|---|---|
| Sujets | 109 volontaires |
| Canaux EEG | 64 (système international 10-10) |
| Fréquence d'échantillonnage | 160 Hz |
| Système d'acquisition | BCI2000 |
| Amplificateur | Non spécifié (électrodes humides Ag/AgCl avec gel) |
| Référence | Lobe d'oreille (référence commune) |
| Terre (ground) | Mastoïde ipsilatérale |
| Impédance cible | < 5 kOhm (standard clinique avec gel) |
| Filtre notch | 60 Hz (enregistré aux USA) |
| Nombre de runs par sujet | 14 |
| Durée totale par sujet | ~28 minutes |
| Format | EDF+ avec annotations T0/T1/T2 |

---

## 2. Les 14 runs — Structure exacte

Chaque sujet réalise exactement 14 runs dans cet ordre :

| Run | Fichier | Type | Durée | Contenu |
|---|---|---|---|---|
| **R01** | S001R01.edf | Baseline yeux ouverts | 60 s | Repos, yeux ouverts, fixation |
| **R02** | S001R02.edf | Baseline yeux fermés | 60 s | Repos, yeux fermés |
| **R03** | S001R03.edf | Mouvement réel (mains) | 125 s | Ouvrir/fermer le poing gauche ou droit |
| **R04** | S001R04.edf | **Imagerie motrice (mains)** | 125 s | **Imaginer** ouvrir/fermer le poing gauche ou droit |
| **R05** | S001R05.edf | Mouvement réel (poings/pieds) | 125 s | Ouvrir/fermer les deux poings ou les deux pieds |
| **R06** | S001R06.edf | **Imagerie motrice (poings/pieds)** | 125 s | **Imaginer** ouvrir/fermer les deux poings ou les deux pieds |
| **R07** | S001R07.edf | Mouvement réel (mains) | 125 s | Répétition de R03 |
| **R08** | S001R08.edf | **Imagerie motrice (mains)** | 125 s | Répétition de R04 |
| **R09** | S001R09.edf | Mouvement réel (poings/pieds) | 125 s | Répétition de R05 |
| **R10** | S001R10.edf | **Imagerie motrice (poings/pieds)** | 125 s | Répétition de R06 |
| **R11** | S001R11.edf | Mouvement réel (mains) | 125 s | Répétition de R03 |
| **R12** | S001R12.edf | **Imagerie motrice (mains)** | 125 s | Répétition de R04 |
| **R13** | S001R13.edf | Mouvement réel (poings/pieds) | 125 s | Répétition de R05 |
| **R14** | S001R14.edf | **Imagerie motrice (poings/pieds)** | 125 s | Répétition de R06 |

**Pour notre projet (classification main gauche / main droite), les runs pertinents sont : R04, R08, R12** (imagerie motrice des mains, 3 sessions).

---

## 3. Structure exacte d'un essai (trial)

Chaque run de tâche (R03–R14) contient exactement **30 annotations** alternant repos et tâche, soit **15 essais** par run.

### Timing extrait des fichiers EDF

```
Essai type (vérifié sur S001R04.edf) :

t=0.00s   [T0] Repos      — durée 4.2 s — écran neutre, croix de fixation
t=4.20s   [T2] Main droite — durée 4.1 s — cible à droite de l'écran
t=8.30s   [T0] Repos      — durée 4.2 s
t=12.50s  [T1] Main gauche — durée 4.1 s — cible à gauche de l'écran
t=16.60s  [T0] Repos      — durée 4.2 s
...
(30 annotations au total, alternance stricte T0/T1 ou T0/T2)
```

### Résumé du timing

| Phase | Code | Durée | Ce qui se passe |
|---|---|---|---|
| **Repos** | T0 | **4.2 secondes** | Écran neutre (croix de fixation). Le sujet se détend. |
| **Tâche** | T1 ou T2 | **4.1 secondes** | Une cible apparaît sur le bord de l'écran. Le sujet exécute ou imagine le mouvement correspondant jusqu'à ce que la cible disparaisse. |

**Durée d'un essai complet** : 4.2 s (repos) + 4.1 s (tâche) = **8.3 secondes**

**Nombre d'essais par run** : 15 (environ 7–8 par classe, pseudo-aléatoire)

**Nombre total d'essais MI mains par sujet** : 15 essais x 3 runs (R04, R08, R12) = **45 essais** (~21 gauche + ~24 droite, variable)

---

## 4. Signification des codes T0, T1, T2

### Pour les runs mains (R03, R04, R07, R08, R11, R12)

| Code | Signification | Cue visuel |
|---|---|---|
| T0 | Repos (relax) | Écran neutre |
| **T1** | **Poing gauche** (ouvrir/fermer ou imaginer) | Cible apparaît à **gauche** de l'écran |
| **T2** | **Poing droit** (ouvrir/fermer ou imaginer) | Cible apparaît à **droite** de l'écran |

### Pour les runs poings/pieds (R05, R06, R09, R10, R13, R14)

| Code | Signification | Cue visuel |
|---|---|---|
| T0 | Repos (relax) | Écran neutre |
| **T1** | **Les deux poings** (ouvrir/fermer ou imaginer) | Cible apparaît en **haut** de l'écran |
| **T2** | **Les deux pieds** (bouger ou imaginer) | Cible apparaît en **bas** de l'écran |

**ATTENTION** : T1 et T2 changent de signification selon le type de run. C'est un piège classique lors de l'extraction des données.

---

## 5. Interface visuelle BCI2000

Le système BCI2000 affiche à l'écran :

### Pendant le repos (T0)
- Fond noir
- Croix de fixation au centre (le sujet fixe le centre)

### Pendant la tâche (T1/T2)
- Une **cible** (rectangle coloré) apparaît sur le bord de l'écran :
  - **Bord gauche** = poing gauche (T1 pour runs mains)
  - **Bord droit** = poing droit (T2 pour runs mains)
  - **Bord haut** = deux poings (T1 pour runs poings/pieds)
  - **Bord bas** = deux pieds (T2 pour runs poings/pieds)
- La cible reste visible pendant **4.1 secondes**
- Le sujet exécute (ou imagine) le mouvement **tant que la cible est visible**
- Quand la cible disparaît → le sujet se détend immédiatement

### Pas de feedback en ligne
Dans le protocole EEGMMIDB original, il n'y a **pas de feedback en temps réel** (pas de curseur contrôlé par le cerveau). C'est un paradigme de **cue-based motor imagery** pur : le sujet voit une instruction et exécute/imagine, sans retour sur sa performance.

---

## 6. Montage 64 canaux — Liste complète

Les 64 électrodes suivent le système international 10-10 (convention de Sharbrough), dans cet ordre exact dans les fichiers EDF :

| Index | Canal | Région | Index | Canal | Région |
|---|---|---|---|---|---|
| 0 | FC5 | Fronto-central G | 32 | F1 | Frontal G |
| 1 | FC3 | Fronto-central G | 33 | Fz | Frontal midline |
| 2 | FC1 | Fronto-central G | 34 | F2 | Frontal D |
| 3 | FCz | Fronto-central mid | 35 | F4 | Frontal D |
| 4 | FC2 | Fronto-central D | 36 | F6 | Frontal D |
| 5 | FC4 | Fronto-central D | 37 | F8 | Frontal D |
| 6 | FC6 | Fronto-central D | 38 | FT7 | Fronto-temporal G |
| 7 | C5 | Central G | 39 | FT8 | Fronto-temporal D |
| 8 | **C3** | **Central G (moteur)** | 40 | T7 | Temporal G |
| 9 | C1 | Central G | 41 | T8 | Temporal D |
| 10 | **Cz** | **Vertex (moteur)** | 42 | T9 | Temporal inf G |
| 11 | C2 | Central D | 43 | T10 | Temporal inf D |
| 12 | **C4** | **Central D (moteur)** | 44 | TP7 | Temporo-pariétal G |
| 13 | C6 | Central D | 45 | TP8 | Temporo-pariétal D |
| 14 | CP5 | Centro-pariétal G | 46 | P7 | Pariétal G |
| 15 | **CP3** | **Centro-pariétal G** | 47 | P5 | Pariétal G |
| 16 | CP1 | Centro-pariétal G | 48 | P3 | Pariétal G |
| 17 | CPz | Centro-pariétal mid | 49 | P1 | Pariétal G |
| 18 | CP2 | Centro-pariétal D | 50 | Pz | Pariétal midline |
| 19 | **CP4** | **Centro-pariétal D** | 51 | P2 | Pariétal D |
| 20 | CP6 | Centro-pariétal D | 52 | P4 | Pariétal D |
| 21 | Fp1 | Préfrontal G | 53 | P6 | Pariétal D |
| 22 | Fpz | Préfrontal mid | 54 | P8 | Pariétal D |
| 23 | Fp2 | Préfrontal D | 55 | PO7 | Pariéto-occipital G |
| 24 | AF7 | Antéro-frontal G | 56 | PO3 | Pariéto-occipital G |
| 25 | AF3 | Antéro-frontal G | 57 | POz | Pariéto-occipital mid |
| 26 | AFz | Antéro-frontal mid | 58 | PO4 | Pariéto-occipital D |
| 27 | AF4 | Antéro-frontal D | 59 | PO8 | Pariéto-occipital D |
| 28 | AF8 | Antéro-frontal D | 60 | O1 | Occipital G |
| 29 | F7 | Frontal G | 61 | Oz | Occipital mid |
| 30 | F5 | Frontal G | 62 | O2 | Occipital D |
| 31 | F3 | Frontal G | 63 | Iz | Inion |

**En gras** : les 8 canaux recommandés pour notre montage sensorimoteur OpenBCI.

### Indices pour l'extraction des canaux (pour le transfer learning)

```
Montage recommandé (8 canaux sensorimoteurs) :
  C3  → index 8
  Cz  → index 10
  C4  → index 12
  FC3 → index 1
  FC4 → index 5
  CP3 → index 15
  CP4 → index 19
  FCz → index 3
```

### Électrodes exclues du système 10-10

Les positions suivantes ne sont **pas** enregistrées dans EEGMMIDB : Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, P10.

---

## 7. Configuration de l'amplificateur BCI2000

| Paramètre | Valeur |
|---|---|
| Mode d'acquisition | Signal analogique |
| Filtrage hardware | Filtre anti-aliasing intégré à l'amplificateur |
| Filtre notch | 60 Hz (USA) — activé pendant l'enregistrement |
| Fréquence d'échantillonnage | 160 Hz |
| Résolution ADC | Non spécifié (typiquement 24 bits pour les amplis EEG cliniques) |
| Référence | Lobe d'oreille (clip) |
| Ground | Mastoïde ipsilatérale |
| Matériau électrodes | Ag/AgCl (argent/chlorure d'argent) avec gel conducteur |
| Impédance cible | < 5 kOhm |

### Préparation des électrodes (protocole BCI2000)

1. Marquer le vertex (Cz) au feutre : milieu entre nasion et inion
2. Placer le bonnet EEG, aligner Cz sur la marque
3. Vérifier : ligne Fz-Cz-Pz sur la ligne médiane, lignes Fp1-Fp2 et O1-O2 horizontales
4. Appliquer du gel conducteur sous chaque électrode
5. Frotter légèrement la peau sous l'électrode avec un bâtonnet en bois (casse la couche sèche de la peau)
6. Mesurer l'impédance de chaque électrode — objectif < 5 kOhm
7. Fixer la référence sur le lobe d'oreille, le ground sur la mastoïde

---

## 8. Instructions données aux sujets

### Baselines (R01, R02)

**R01 — Yeux ouverts (60 s) :**
> "Restez assis confortablement, les yeux ouverts. Fixez un point devant vous. Ne bougez pas."

**R02 — Yeux fermés (60 s) :**
> "Fermez les yeux. Restez détendu et immobile pendant une minute."

### Mouvement réel — Mains (R03, R07, R11)

> "Une cible va apparaître sur le côté gauche ou droit de l'écran. Quand la cible apparaît à gauche, ouvrez et fermez votre poing gauche de manière répétée jusqu'à ce que la cible disparaisse. Quand la cible apparaît à droite, ouvrez et fermez votre poing droit. Quand il n'y a pas de cible, détendez-vous."

### Imagerie motrice — Mains (R04, R08, R12) — LE PLUS IMPORTANT POUR NOUS

> "Une cible va apparaître sur le côté gauche ou droit de l'écran. Quand la cible apparaît à gauche, **imaginez** que vous ouvrez et fermez votre poing gauche, sans bouger réellement. Quand la cible apparaît à droite, **imaginez** que vous ouvrez et fermez votre poing droit. Quand il n'y a pas de cible, détendez-vous."

### Mouvement réel — Poings/Pieds (R05, R09, R13)

> "Une cible va apparaître en haut ou en bas de l'écran. Quand la cible apparaît en haut, ouvrez et fermez les deux poings de manière répétée. Quand la cible apparaît en bas, bougez les deux pieds. Quand il n'y a pas de cible, détendez-vous."

### Imagerie motrice — Poings/Pieds (R06, R10, R14)

> "Une cible va apparaître en haut ou en bas de l'écran. Quand la cible apparaît en haut, **imaginez** que vous ouvrez et fermez les deux poings. Quand la cible apparaît en bas, **imaginez** que vous bougez les deux pieds. Quand il n'y a pas de cible, détendez-vous."

---

## 9. Sujets à exclure

Shuqfa et al. (2024) ont analysé l'intégralité du dataset et identifié **6 sujets problématiques** :

| Sujet | Problème |
|---|---|
| S088 | Nombre d'essais incohérent |
| S089 | Anomalie d'enregistrement |
| S092 | Nombre d'essais incohérent |
| S100 | Nombre d'essais incohérent |
| S104 | Nombre d'essais incohérent |
| S106 | Anomalie d'enregistrement |

**Sujets exploitables : 103 sur 109.**

Référence : Shuqfa et al., "Increasing accessibility to a large brain-computer interface dataset", Data in Brief, 2024.

---

## 10. Comment reproduire ce protocole avec OpenBCI

### 10.1 Différences matérielles à compenser

| Paramètre | PhysioNet (original) | Notre setup (OpenBCI) | Action requise |
|---|---|---|---|
| Canaux | 64 (10-10 complet) | 8 (sensorimoteurs) | Migrer vers montage C3/Cz/C4/FC3/FC4/CP3/CP4/FCz |
| Fs | 160 Hz | 250 Hz | Rééchantillonner à 160 Hz pour le transfer learning, OU suréchantillonner PhysioNet à 250 Hz |
| Électrodes | Humides Ag/AgCl + gel | Sèches ThinkPulse | Bien dégager les cheveux, accepter un SNR inférieur |
| Impédance | < 5 kOhm | 50–500 kOhm (sèches) | Faire le maximum pour réduire : contact direct cuir chevelu |
| Référence | Lobe d'oreille | SRB (clip d'oreille) | Compatible, vérifier le bon contact |
| Notch | 60 Hz (USA) | 50 Hz (France) | Appliquer notch 50 Hz sur nos données, 60 Hz sur PhysioNet |
| Durée tâche | 4.1 s | 4.1 s | Reproduire exactement |
| Durée repos | 4.2 s | 4.2 s | Reproduire exactement |
| Essais/run | 15 | 15 | Reproduire exactement |
| Runs MI mains | 3 (R04, R08, R12) | 3 minimum | Même structure |

### 10.2 Protocole de reproduction — Pas à pas

#### Avant la session

1. **Montage du casque** : suivre le protocole de la section 2 de `protocole-berger.md` (cheveux, contact, clip d'oreille)

2. **Mapping des électrodes** : s'assurer que les 8 canaux sont positionnés sur :
   ```
   CH1 → C3    CH5 → CP3
   CH2 → Cz    CH6 → CP4
   CH3 → C4    CH7 → FCz
   CH4 → FC3   CH8 → FC4
   ```
   Utiliser `python main.py` → choix **6** (Mapping du casque) pour configurer et sauvegarder.

3. **Vérification du signal** : `python main.py` → choix **2** (Temps réel). Activer bandpass + notch 50 Hz + CAR. Vérifier que tous les canaux montrent un signal propre (30–80 µV, pas de saturation, pas de 50 Hz dominant).

#### Session d'enregistrement (reproduit le protocole PhysioNet)

La session complète comprend :

```
PHASE 1 — BASELINE YEUX OUVERTS (60 s)
═══════════════════════════════════════
  → Reproduit R01
  Enregistrer (R) → Sujet + "Yeux fermés" → OK
  M → "Ouvrir les yeux"
  Le sujet fixe un point devant lui, immobile, 60 secondes.
  R pour arrêter.

PHASE 2 — BASELINE YEUX FERMÉS (60 s)
═══════════════════════════════════════
  → Reproduit R02
  Enregistrer (R) → Sujet + "Yeux fermés" → OK
  M → "Fermer les yeux"
  Le sujet ferme les yeux, détendu, 60 secondes.
  R pour arrêter.

PAUSE (1-2 min) — Le sujet peut bouger.

PHASE 3, 4, 5 — IMAGERIE MOTRICE MAINS (3 x 125 s)
════════════════════════════════════════════════════
  → Reproduit R04, R08, R12
  Pour chaque run :
    Enregistrer (R) → Sujet + "Mouvement des mains" → OK
    Suivre le protocole de 15 essais décrit ci-dessous.
    R pour arrêter.
  Pause de 1-2 min entre chaque run.
```

#### Déroulement d'un run d'imagerie motrice (15 essais)

**Ce qu'il faudra implémenter** (stimulus visuel automatisé, comme `ssvep_stimulus.py` mais pour MI) :

```
Pour chaque essai (15 essais par run, alternant gauche/droite pseudo-aléatoire) :

1. REPOS (4.2 s)
   Écran : fond gris, croix de fixation au centre
   Sujet : se détend, ne pense à rien

2. CUE + IMAGERIE (4.1 s)
   Écran : flèche vers la GAUCHE ou vers la DROITE
   Sujet : imagine ouvrir/fermer le poing correspondant
           (sans bouger réellement !)
   Marqueur automatique : "T1" (gauche) ou "T2" (droite)

→ Retour au repos, essai suivant
```

**En attendant le stimulus automatisé**, on peut faire les essais manuellement avec l'opérateur qui dit "gauche" / "droite" à voix haute et pose les marqueurs avec M, mais le timing sera moins précis.

### 10.3 Nombre d'essais récapitulatif

| Donnée | PhysioNet | Notre reproduction |
|---|---|---|
| Essais MI mains par run | 15 (~7-8 par classe) | 15 |
| Runs MI mains | 3 (R04, R08, R12) | 3 minimum |
| **Total essais MI mains** | **~45** (~21 G + ~24 D) | **~45** |
| Durée MI mains | 3 x 125 s = 6.25 min | 6.25 min |
| Durée totale session | ~28 min (14 runs) | ~15 min (baselines + 3 runs MI) |

---

## 11. Prétraitement recommandé (aligné PhysioNet)

Pour que le transfer learning fonctionne, appliquer **exactement le même prétraitement** sur les deux sources :

| Étape | PhysioNet | Nos données OpenBCI |
|---|---|---|
| 1. Rééchantillonnage | Natif 160 Hz | Rééchantillonner 250 Hz → 160 Hz |
| 2. Sélection canaux | Extraire indices 8,10,12,1,5,15,19,3 | Tous les 8 canaux (déjà sensorimoteurs) |
| 3. Filtre passe-bande | Butterworth ordre 4, **4–40 Hz**, zero-phase (filtfilt) | Idem |
| 4. Filtre notch | 60 Hz (Q=30) | 50 Hz (Q=30) |
| 5. CAR | Appliquer (remplace la ref. moyenne de PhysioNet) | Appliquer |
| 6. Extraction époques | 0.5 s à 4.0 s après onset T1/T2 (3.5 s utiles) | Idem |
| 7. Rejection | Amplitude crête-à-crête > 100 µV → rejet | Idem |
| 8. Normalisation | Z-score par canal par époque (moyenne=0, variance=1) | Idem |

**Note : le bandpass doit être 4–40 Hz (pas 1–50 Hz comme dans le code actuel).** La borne basse à 4 Hz supprime les dérives, la borne haute à 40 Hz supprime l'EMG et le secteur.

---

## 12. Vérification rapide sur PhysioNet

Pour vérifier que le pipeline fonctionne avant de passer aux données OpenBCI :

```bash
python main.py
```

1. Choix **3** → Télécharger les données (sujets 1 2 3)
2. Choix **1** → Analyse offline → Sujet `S001` → Run **3** (R04, imagerie motrice)
3. Sélectionner les canaux : `C3 CZ C4`
4. Regarder la PSD : on doit voir une activité dans la bande 8–30 Hz

Pour une analyse plus poussée, charger dans Python :

```python
import mne
mne.set_log_level('WARNING')
raw = mne.io.read_raw_edf('data/S001/S001R04.edf', preload=True, verbose=False)

# Voir les annotations
for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
    print(f't={onset:7.2f}s  dur={dur:.1f}s  [{desc}]')

# Extraire un essai T1 (main gauche) : 0.5s à 4.0s après onset
# Premier T1 est à t=12.50s
# → segment 13.0s à 16.5s
```

---

## Annexe — Résumé en une page

```
PROTOCOLE PHYSIONET EEGMMIDB — RÉSUMÉ
══════════════════════════════════════

DATASET : 109 sujets, 64 canaux, 160 Hz, BCI2000, électrodes humides

14 RUNS PAR SUJET :
  R01 : Baseline yeux ouverts (60 s)
  R02 : Baseline yeux fermés (60 s)
  R03/R07/R11 : Mouvement réel mains (125 s, 15 essais)
  R04/R08/R12 : Imagerie motrice mains (125 s, 15 essais) ← NOTRE CIBLE
  R05/R09/R13 : Mouvement réel poings/pieds (125 s, 15 essais)
  R06/R10/R14 : Imagerie motrice poings/pieds (125 s, 15 essais)

TIMING D'UN ESSAI :
  Repos T0 : 4.2 s (écran neutre)
  Tâche T1/T2 : 4.1 s (cible gauche ou droite)

CODES :
  Runs mains : T1 = poing gauche, T2 = poing droit
  Runs pieds : T1 = deux poings, T2 = deux pieds
  ⚠ T1/T2 changent de sens selon le run !

CANAUX SENSORIMOTEURS (indices dans les EDF) :
  C3=8, Cz=10, C4=12, FC3=1, FC4=5, CP3=15, CP4=19, FCz=3

SUJETS À EXCLURE : S088, S089, S092, S100, S104, S106

PRÉTRAITEMENT : 4–40 Hz, notch 60 Hz, CAR, reject > 100 µV
ÉPOQUES : 0.5 s à 4.0 s après onset T1/T2 (3.5 s)
```

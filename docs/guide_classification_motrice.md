# Guide de classification motrice — EEG OpenBCI Cyton

## Pré-requis matériel

- Casque OpenBCI Ultracortex Mark IV avec carte Cyton (8 canaux, 250 Hz)
- Gel conducteur sur les 8 électrodes + clip earlobe sur le pin **SRB2**
- Câble USB connecté au Mac

### Placement des électrodes (preset Motor Imagery)

| Pin Cyton | Electrode | Rôle |
|-----------|-----------|------|
| CH1 | C3 | Moteur primaire (main droite) |
| CH2 | FC1 | Prémoteur gauche |
| CH3 | C4 | Moteur primaire (main gauche) |
| CH4 | O1 | Occipital gauche |
| CH5 | Cz | Moteur midline (pieds) |
| CH6 | O2 | Occipital droit |
| CH7 | FC2 | Prémoteur droit |
| CH8 | Pz | Pariétal midline |

---

## Tâche 2 — Imagerie main gauche vs droite

### 1. Préparation

- Pièce calme, lumière tamisée, écran à ~60 cm
- Posture : assis, mains posées sur les cuisses, pieds à plat
- Durée : ~4 min par run, faire **minimum 3 runs**

### 2. Lancer l'application

```bash
python main.py
```

### 3. Connexion et vérification du signal

1. Aller dans **Acquisition**
2. Sélectionner le port série, cliquer **Connecter**
3. Attendre **10-15 secondes** que le signal se stabilise
4. Vérifier les 8 indicateurs de canal :
   - Vert = signal OK
   - Jaune = canal plat (électrode déconnectée ou mauvais contact)
   - Rouge = raillé (saturation ADC, vérifier électrode et référence SRB2)
5. Si >= 6 canaux rouges : **la référence SRB2 (earlobe) est débranchée**
6. Cliquer **"Détection automatique"** pour optimiser le gain

### 4. Configuration du protocole

1. Protocole → sélectionner **"Tâche 2 — Imagerie main G/D (R04)"**
2. Les paramètres se remplissent automatiquement :
   - 15 essais par classe (30 total)
   - 2s baseline, 4s cue, 2s repos
3. Le sujet et run label sont auto-incrémentés

### 5. Enregistrement

Cliquer **"Démarrer l'enregistrement"**. Une fenêtre plein écran affiche les cues :

| Affichage | Durée | Action |
|-----------|-------|--------|
| ✛ (croix) | 2s | Relax total, fixer la croix, ne penser à rien |
| ← Imagerie G | 4s | Imaginer serrer fort le poing GAUCHE |
| ✛ (croix) | 2s | Relax |
| → Imagerie D | 4s | Imaginer serrer fort le poing DROIT |

**Règles strictes pendant les 4s de cue :**
- Ne rien bouger : ni les mains, ni la mâchoire, ni les yeux
- Fixer le texte à l'écran
- Imaginer le mouvement de façon **kinesthésique** (sentir la contraction), pas visuelle
- Même intensité à chaque essai

**Entre les runs :**
- Pause 2-3 minutes, boire de l'eau
- Ne pas retirer le casque
- Relancer un enregistrement (le run label s'incrémente : R01, R02, R03)

### 6. Fichiers générés

Les EDF sont sauvegardés dans :
```
data/custom/{SUJET_ID}/{SUJET_ID}R01.edf
data/custom/{SUJET_ID}/{SUJET_ID}R02.edf
data/custom/{SUJET_ID}/{SUJET_ID}R03.edf
```

---

## Tâche 4 — Imagerie 2 mains vs 2 pieds

Même protocole que la Tâche 2, sauf :
- Sélectionner **"Tâche 4 — Imagerie 2 mains / 2 pieds (R06)"**
- Cues : "↑↑ Imagerie mains" / "↓↓ Imagerie pieds"
- **Généralement plus facile** (~71% vs ~63% sur PhysioNet) car les zones cérébrales sont plus éloignées

---

## Analyse des résultats

### Commandes de classification

```bash
# Run par run (identifier un mauvais run)
python -m src.scripts.classify_custom --subject SUJET_ID --runs 1 --task left_vs_right
python -m src.scripts.classify_custom --subject SUJET_ID --runs 2 --task left_vs_right
python -m src.scripts.classify_custom --subject SUJET_ID --runs 3 --task left_vs_right

# Tous les runs combinés (plus fiable)
python -m src.scripts.classify_custom --subject SUJET_ID --runs 1,2,3 --task left_vs_right

# Sauvegarder le modèle entraîné
python -m src.scripts.classify_custom --subject SUJET_ID --runs 1,2,3 --task left_vs_right --save-model

# Pour hands vs feet
python -m src.scripts.classify_custom --subject SUJET_ID --runs 1,2,3 --task hands_vs_feet
```

### Lecture de la sortie

Exemple :
```
  Epochs     : {'T1': 42, 'T2': 43} (rejetes: 5)
  CV accuracy: 68.2% +/- 8.1%
  Par fold   : ['72.2%', '61.1%', '66.7%', '77.8%', '63.3%']
```

| Métrique | Signification |
|----------|---------------|
| **Epochs rejetés** | Nombre d'epochs supprimées car amplitude > 500 µV. Si > 20% du total, le contact était mauvais |
| **CV accuracy** | Accuracy moyenne en cross-validation 5-fold. C'est le chiffre principal |
| **Std accuracy** | Ecart-type entre les folds. Si > 15%, le résultat est instable |
| **Par fold** | Si un fold est très bas, un run est probablement de mauvaise qualité |

### Échelle d'interprétation

| Accuracy | Verdict |
|----------|---------|
| < 55% | Hasard — signal trop bruité ou imagerie insuffisante |
| 55-65% | Faible mais au-dessus du hasard, le signal moteur existe |
| 65-75% | Bon résultat pour de l'imagerie motrice sur 8 canaux |
| 75-85% | Très bon — signal propre et concentration solide |
| > 85% | Excellent — rare en imagerie motrice |

### Confusion matrix

```
               pred T1  pred T2
    vrai T1      12       3      ← gauche bien classé (80%)
    vrai T2       5      10      ← droite plus confus (67%)
```

- Diagonale forte = bonne classification
- Si une ligne est faible, le modèle confond ce côté
- Solution : se concentrer davantage sur l'imagerie du côté faible

### Diagnostic si mauvais résultats (< 55%)

1. **Trop d'epochs rejetés** → refaire avec meilleur contact gel
2. **Un run très différent des autres** → le supprimer et refaire
3. **Tout à ~50%** → essayer `hands_vs_feet` (Tâche 4), plus facile
4. **Std très haute** → ajouter des runs supplémentaires
5. **Signal raillé pendant enregistrement** → vérifier SRB2 et gain

---

## Transfer learning (PhysioNet → custom)

Entraîner un modèle sur un bon sujet PhysioNet, puis le tester sur ses propres données :

```bash
# 1. Entraîner sur un sujet PhysioNet performant
python -m src.scripts.classify_physionet --subjects 29 --task left_vs_right --save-model

# 2. Tester sur ses données custom
python -m src.scripts.classify_custom --subject SUJET_ID --runs 1,2,3 \
  --transfer models/S029_left_vs_right_csp_lda.pkl
```

**Interprétation** : L'accuracy transfer sera souvent inférieure à la cross-validation car les patterns CSP sont spécifiques à chaque cerveau. Si c'est comparable, le modèle généralise bien.

### Meilleurs sujets PhysioNet pour le transfer

**Left vs Right** : S029 (97.8%), S062 (97.8%), S094 (95.6%), S002 (93.3%)
**Hands vs Feet** : S001 (100%), S035 (97.8%), S042 (97.8%), S062 (97.8%)

---

## Pipeline technique (référence)

```
EDF → Load + Merge runs → Bandpass 8-30 Hz (+ notch 50 Hz)
    → Epoch [0.5s, 3.5s] après onset → Rejet artefacts (> 500 µV)
    → CSP (6 composantes, régularisation Ledoit-Wolf) → LDA
    → Cross-validation stratifiée 5-fold
```

Le modèle sauvegardé (`.pkl`) contient le pipeline complet CSP+LDA et peut être réutilisé pour prédire de nouvelles epochs via `EEGClassificationService.predict_epoch()`.

# Interface Cerveau-Machine par

# Imagerie Motrice EEG :

## Construction d’un Modèle de Classification

Rapport de Recherche _— Revue de littérature, Méthodes & Guide d’implémentation_

### Projet de Recherche 3A — EEG Object Control

```
École Nationale Supérieure des Mines de Saint-Étienne
Encadrant : David Moreau (david.moreau@emse.fr)
Mars 2026
```

## Table des matières

## Table des matières

- Table des matières
- I. Résumé
- II. Introduction
   - A. Contexte et motivation
   - B. Bases neurophysiologiques de l’imagerie motrice
   - C. Placement des électrodes pour l’imagerie motrice
- III. Jeux de données de référence
   - A. PhysioNet EEGMMIDB....................................................................................................................
   - B. Jeux de données BCI Competition IV
- IV. Méthodes de classification
   - A. Approche classique : CSP + LDA......................................................................................................
   - B. Deep Learning : EEGNet
   - C. Architectures récentes (2024–2025)
- V. Transfer Learning : de PhysioNet à OpenBCI
   - A. Pourquoi le transfer learning est nécessaire........................................................................................
   - B. Pipeline proposé
- VI. Illettrisme BCI : un défi connu
- VII. Bonnes pratiques de prétraitement
- VIII. Perspectives et améliorations
   - A. Court terme (dans le cadre de ce projet)
   - B. Moyen terme (itérations futures du projet)
   - C. Long terme (perspectives de recherche)
- IX. Conclusion
- Références


## I. Résumé

Ce rapport fournit une base de recherche complète pour la construction d’un système de classification par
Interface Cerveau-Machine (ICM, ou BCI en anglais) fondé sur l’Imagerie Motrice (IM) à partir d’un casque
EEG grand public (OpenBCI). L’objectif est de décoder les mouvements imaginés des mains à partir de
signaux électroencéphalographiques enregistrés au-dessus du cortex sensorimoteur, puis de les traduire en
commandes de contrôle pour des dispositifs externes tels que des prothèses robotiques, des interfaces de jeu
ou des technologies d’assistance. Nous passons en revue les bases neurophysiologiques de l’imagerie
motrice, évaluons les jeux de données de référence (PhysioNet EEGMMIDB), analysons les algorithmes de
classification de l’état de l’art — des Common Spatial Patterns aux architectures de deep learning (EEGNet)
— et proposons un pipeline concret de transfer learning pour combler l’écart entre les données publiques et
les enregistrements réels OpenBCI. Le placement des électrodes, les stratégies de prétraitement et les pièges
connus, y compris l’illettrisme BCI, sont discutés.

Mots-clés : Imagerie Motrice, Interface Cerveau- _Machine, EEG, Désynchronisation Liée à l’Événement,_
Common Spatial Patterns, EEGNet, Transfer Learning, PhysioNet, OpenBCI

## II. Introduction

### A. Contexte et motivation

Les Interfaces Cerveau-Machine (ICM) établissent un canal de communication direct entre le cerveau humain
et des dispositifs externes en décodant l’activité neurale, contournant les voies neuromusculaires classiques
[1]. Parmi les différents paradigmes BCI, l’Imagerie Motrice (IM) est particulièrement attractive pour les
applications de contrôle actif car elle repose sur des signaux cérébraux endogènes — l’utilisateur génère
volontairement le signal de contrôle en imaginant un mouvement, sans nécessiter de stimulus externe [2].
Cela rend l’IM-BCI adaptée au contrôle de prothèses, fauteuils roulants, bras robotiques, interfaces de jeu et
dispositifs de communication assistée.

Le présent projet vise à construire un système IM-BCI fonctionnel utilisant un équipement EEG grand public
(OpenBCI avec le casque Ultracortex Mark IV). L’objectif spécifique est de classifier l’imagination de
mouvements de la main gauche vs. main droite à partir d’enregistrements EEG 8 canaux et d’utiliser ces
classifications pour contrôler des objets en temps réel.

### B. Bases neurophysiologiques de l’imagerie motrice

Lorsqu’une personne imagine un mouvement (par exemple, serrer le poing droit), les mêmes aires corticales
s’activent que lors de l’exécution réelle du mouvement, en particulier le cortex moteur primaire (M1) et l’aire
motrice supplémentaire (SMA) [3]. Cette activation produit des changements mesurables dans l’activité
oscillatoire de l’EEG :

**Désynchronisation liée à l’événement (ERD) :** Une diminution de la puissance dans les bandes de fréquence
mu (8–13 Hz) et bêta (13–30 Hz) au-dessus du cortex sensorimoteur controlatéral au mouvement imaginé.
L’ERD reflète une augmentation du traitement cortical et de l’engagement neural dans la tâche motrice [4].

**Synchronisation liée à l’événement (ERS) :** Une augmentation de la puissance dans les mêmes bandes de
fréquence, typiquement observée sur l’hémisphère ipsilatéral (le côté qui NE correspond PAS à la main
imaginée) ou sous forme de rebond bêta post-mouvement. L’ERS reflète un état de repos cortical ou une
inhibition active [4].

Point critique pour la discrimination main gauche vs. main droite : imaginer un mouvement de la main droite
produit un ERD sur l’électrode C3 (hémisphère gauche, controlatéral), tandis qu’imaginer un mouvement de


la main gauche produit un ERD sur l’électrode C4 (hémisphère droit, controlatéral). Cette dominance
controlatérale est la caractéristique principale exploitée par les classifieurs MI-BCI [5].

```
⚠ Le rythme mu (aussi appelé rythme sensorimoteur ou SMR) est distinct du rythme alpha occipital.
Bien que les deux se situent dans la bande 8 – 13 Hz, ils proviennent d’aires corticales différentes et ont
des significations fonctionnelles différentes. Le rythme mu est généré par le cortex sensorimoteur et
modulé par l’action et l’imagerie motrice, tandis que l’alpha est généré par le cortex visuel e t modulé
par l’entrée visuelle.
```
### C. Placement des électrodes pour l’imagerie motrice

Le positionnement des électrodes est critique pour les performances MI-BCI. L’approche standard consiste
à concentrer les électrodes au-dessus du cortex sensorimoteur selon le système international 10-20/10-10 [6].
Pfurtscheller et al. ont démontré que les électrodes C3 et C4, situées au-dessus des zones de représentation
de la main dans M1, fournissent la meilleure discrimination entre les différentes tâches d’imagerie motrice
chez la majorité des sujets [5]. Le montage 8 canaux recommandé pour ce projet est :

```
Électrode Région cérébrale Rôle en MI-BCI Priorité
C3 Cortex moteur gauche (M1) ERD lors de l’IM main droite Essentiel
C4 Cortex moteur droit (M1) ERD lors de l’IM main
gauche
```
```
Essentiel
```
```
Cz Vertex / M1 pieds IM des pieds (3ème classe) Essentiel
FC3, FC4 Cortex prémoteur Préparation / intention
motrice
```
```
Élevée
```
```
CP3, CP4 Cortex somatosensoriel Feedback sensoriel lors de
l’IM
```
```
Élevée
```
```
FCz Aire motrice supplémentaire Planification motrice ; classe
pieds
```
```
Recommandée
```
```
Tableau 1. Montage 8 canaux recommandé pour l’IM - BCI. Cette configuration permet le filtrage
Laplacien spatial autour de C3/C4 pour un meilleur rapport signal/bruit.
⚠ Ce montage remplace le placement par défaut à couverture large (O1, O2, P3, P4, C3, C4, F3, F4).
Les électrodes occipitales sont inutiles pour l’IM et gaspillent de la capacité. Les 8 canaux doivent être
concentrés sur la bande sensorimotrice.
```

## III. Jeux de données de référence

### A. PhysioNet EEGMMIDB....................................................................................................................

Le PhysioNet EEGMMIDB [7] est le plus grand jeu de données MI-EEG disponible publiquement. Il
comprend plus de 1 500 enregistrements provenant de 109 volontaires, acquis avec 64 électrodes à 160 Hz
via le système BCI2000 [8]. Chaque sujet a réalisé 14 sessions expérimentales incluant des baselines (yeux
ouverts/fermés) et quatre tâches motrices/d’imagerie : mouvement réel et imaginé d’ouverture/fermeture du
poing (gauche/droit), mouvement réel et imaginé des deux poings ou des deux pieds [7].

Structure des tâches

Chaque session de deux minutes contient environ 14 essais. Chaque essai comprend une période de tâche de
4,1 secondes indiquée par des indices visuels (T1 pour le poing gauche ou les deux poings, T2 pour le poing
droit ou les deux pieds), entrecoupée de périodes de repos de 4,2 secondes (T0). Les sessions pertinentes pour
la classification IM sont : Runs 4, 8, 12 (IM poing gauche/droit) et Runs 6, 10, 14 (IM deux poings/pieds)
[7].

Points de vigilance sur la qualité des données

Shuqfa et al. [9] ont nettoyé ce jeu de données et identifié 6 sujets avec des enregistrements anormaux à
exclure, résultant en 103 sujets exploitables. Ils fournissent une version nettoyée en format MATLAB et CSV
sur Mendeley Data. Points de vigilance essentiels :

1. Sujets à exclure : Les sujets S088, S092, S100, S104 ont des nombres d’essais incohérents. Les sujets
S089 et S106 présentent des anomalies d’enregistrement [9].
**2. Fréquence d’échantillonnage :** 160 Hz, inférieure à de nombreux systèmes modernes. Si votre OpenBCI
enregistre à 250 Hz, vous devez soit sous-échantillonner vos données à 160 Hz, soit sur-échantillonner
PhysioNet à 250 Hz avant le transfer learning.
3. Annotations : Les événements sont codés T0 (repos), T1 (poing gauche ou deux poings selon la session),
T2 (poing droit ou deux pieds). La signification de T1/T2 change entre les types de sessions — vérifiez
soigneusement quelles sessions correspondent à quelle tâche.
4. Extraction des canaux : Les 64 canaux suivent le système 10-10. Pour correspondre à votre montage
OpenBCI 8 canaux, extrayez : C3 (indice 7), Cz (indice 9), C4 (indice 11), FC3 (indice 5), FC4 (indice 39),
CP3 (indice 42), CP4 (indice 46), FCz (indice 4). Vérifiez les indices par rapport à la carte officielle des
électrodes.
5. Schéma de référence : PhysioNet utilise une référence moyenne commune. Votre OpenBCI utilisera
probablement une référence sur le lobe d’oreille. Cette différence contribue au décalage de domaine et doit
être traitée lors du prétraitement ou de l’alignement.

### B. Jeux de données BCI Competition IV

Dataset 2a [10] : IM 4 classes (main gauche, main droite, pieds, langue), 22 canaux EEG, 9 sujets, 250 Hz.
Enregistré par le groupe BCI de Graz. C’est le benchmark principal pour les algorithmes MI multi-classes.

Dataset 2b [10] : IM 2 classes (main gauche/droite), 3 canaux bipolaires (C3, Cz, C4), 9 sujets, 5 sessions
incluant du feedback en ligne. C’est le benchmark le plus pertinent pour les configurations à nombre réduit
de canaux comme la vôtre.


## IV. Méthodes de classification

### A. Approche classique : CSP + LDA......................................................................................................

Les Common Spatial Patterns (CSP) [11] constituent la méthode d’extraction de caractéristiques la plus
établie pour l’IM-BCI. Le CSP trouve des filtres spatiaux qui maximisent la variance d’une classe tout en
minimisant celle de l’autre. Cela exploite directement les patterns latéralisés d’ERD/ERS : pour l’IM main
gauche vs. droite, le CSP apprend des filtres qui accentuent la différence de puissance entre les régions C3 et
C4.

Le pipeline standard est : (1) filtrer l’EEG en passe-bande 8–30 Hz, (2) extraire l’époque MI (typiquement
0,5–4 s après l’indice), (3) calculer les filtres spatiaux CSP, (4) extraire les caractéristiques log-variance des
signaux filtrés, et (5) classifier avec l’Analyse Discriminante Linéaire (LDA) [11]. Ce pipeline est simple,
interprétable et sert de baseline essentielle.

Filter Bank CSP (FBCSP) [12] : Une amélioration qui applique le CSP à plusieurs sous-bandes (ex : 4–8,
8 – 12, 12–16, ..., 36–40 Hz) et sélectionne les caractéristiques les plus discriminantes. Le FBCSP a remporté
la BCI Competition IV avec des valeurs kappa de 0,569 (4 classes) et 0,600 (2 classes) [12]. Il répond à la
limite que la bande de fréquence optimale varie selon les sujets.

### B. Deep Learning : EEGNet

EEGNet [13] est un réseau de neurones convolutif (CNN) compact spécifiquement conçu pour les BCI basées
sur l’EEG. Il encapsule les concepts fondamentaux du traitement du signal MI-BCI (filtrage temporel, filtrage
spatial, extraction de caractéristiques) dans une architecture entraînable de bout en bout. EEGNet utilise trois
innovations clés :

1. Convolution temporelle : Apprend des filtres spécifiques en fréquence (analogue au filtrage passe-bande).
2. Convolution en profondeur (depthwise) : Apprend des filtres spatiaux pour chaque carte de
caractéristiques temporelle (analogue au CSP).
3. Convolution séparable : Combine les caractéristiques temporelles et spatiales efficacement avec
beaucoup moins de paramètres que les convolutions standard.

Sur le jeu de données PhysioNet avec 64 canaux, EEGNet atteint 82,43% de précision pour l’IM 2 classes en
validation globale (cross-sujet) [14]. Avec une réduction à 8 canaux, la précision diminue modérément mais
reste supérieure au hasard lorsque combinée avec du transfer learning spécifique au sujet [14]. Le code est
disponible publiquement à : github.com/vlawhern/arl-eegmodels [13].

### C. Architectures récentes (2024–2025)

Plusieurs architectures plus récentes améliorent EEGNet et montrent des progrès sur les benchmarks
standards :

```
Modèle Innovation clé Préc. BCI IV-2a Référence
EEGNet Conv. depthwise + séparable ~77% Lawhern et al., 2018 [13]
ShallowConvNet Architecture inspirée du CSP ~75% Schirrmeister et al., 2017
[15]
ATCNet Attention + TCN + CNN ~83% Altaheri et al., 2022 [16]
CTNet CNN + encodeur Transformer ~84% Zhao et al., 2024 [17]
CIACNet Double branche + conv.
attention
```
#### ~85% 2025 [19]


Tableau 2. Comparaison des précisions sur BCI Competition IV-2a (IM 4 classes). Pour les tâches 2
classes, tous les modèles atteignent des performances nettement supérieures.
⚠ Recommandation pour ce projet : commencez par CSP+LDA comme baseline, puis implémentez
EEGNet. Les architectures plus complexes (ATCNet, CTNet) offrent des gains marginaux mais
augmentent considérablement la complexité. EEGNet offre le meilleur compromis
performance/simplicité pour un projet à 3 personnes avec des données limitées.


## V. Transfer Learning : de PhysioNet à OpenBCI

### A. Pourquoi le transfer learning est nécessaire........................................................................................

Entraîner un modèle de deep learning from scratch nécessite des centaines d’essais labellisés par sujet, soit
environ 45–60 minutes d’enregistrement par personne — impraticable pour un projet de recherche. Le
transfer learning résout ce problème en pré-entraînant un modèle sur un grand jeu de données public
(PhysioNet, 103 sujets) puis en l’ajustant (fine-tuning) avec un petit nombre de données de vos propres sujets
(20–40 essais) [20].

Cependant, transférer entre jeux de données introduit un décalage de domaine (domain shift) provenant de
trois sources : (1) les différences matérielles (BCI2000 64 canaux électrodes humides vs. OpenBCI 8 canaux
électrodes sèches), (2) la variabilité inter-sujets (chaque cerveau est différent), et (3) les différences
environnementales (laboratoire blindé vs. votre salle d’enregistrement). Entre 15 et 30% des utilisateurs BCI
ne parviennent pas à produire des patterns MI classifiables — un phénomène appelé « illettrisme BCI »
[21][22].

### B. Pipeline proposé

Le pipeline complet se compose de cinq étapes :

Étape 1 **_—_** Préparation des données (PhysioNet)

Télécharger PhysioNet EEGMMIDB. Extraire uniquement les sessions MI (4, 8, 12 pour poing gauche/droit).
Sélectionner les 8 canaux correspondant à votre montage OpenBCI (C3, Cz, C4, FC3, FC4, CP3, CP4, FCz).
Appliquer un filtre passe-bande 4–40 Hz (Butterworth ordre 4). Segmenter les époques de 0,5 s à 4,0 s après
l’apparition de l’indice. Rejeter les époques avec une amplitude > 100 μV. Normaliser chaque époque à
moyenne nulle et variance unitaire. Cela donne environ 21 essais × 2 classes × 3 sessions = 63 essais par
sujet et par classe, sur 103 sujets.

Étape 2 **_—_** Alignement Euclidien (adaptation de domaine)

Appliquer l’Euclidean Alignment (EA) [23] pour réduire la variabilité inter-sujets et inter-dispositifs. Pour
chaque sujet/session, calculer la moyenne arithmétique des matrices de covariance des essais, puis blanchir
tous les essais par cette moyenne. Cette opération linéaire simple aligne les distributions de données entre
sujets et améliore significativement le transfert cross-sujet sans nécessiter de données labellisées du domaine
cible [23].

Étape 3 **_—_** Pré-entraînement (cross-sujet)

Entraîner un modèle EEGNet sur les données PhysioNet en validation croisée 5-fold sur les sujets
(entraînement sur 80% des sujets, validation sur 20%). Utiliser l’optimiseur Adam avec un taux
d’apprentissage de 10⁻³, la perte cross-entropy catégorielle, une taille de batch de 64, pendant 300– 500
époques avec arrêt précoce (patience = 50). Performance attendue : 70–82% pour l’IM 2 classes avec 8
canaux [14].

Étape 4 **_—_** Enregistrement de vos propres données (OpenBCI)

Enregistrer chaque membre de l’équipe en utilisant le même paradigme que PhysioNet : IM gauche/droite
avec indice, essais de 4 secondes, repos inter-essais de 2–3 secondes. Collecter un minimum de 40 essais
par classe (80 au total par sujet, environ 15 minutes d’enregistrement). Appliquer le même filtre passe-bande
et la même segmentation qu’à l’étape 1. Appliquer l’EA en utilisant uniquement vos données de repos comme
référence.

Étape 5 **_—_** Fine-tuning (spécifique au sujet)


Geler les deux premiers blocs convolutifs de l’EEGNet pré-entraîné (convolutions temporelles et depthwise
qui ont appris les patterns ERD/ERS génériques). Ré-entraîner uniquement le bloc de convolution séparable
et la couche de classification sur vos données labellisées, en validation croisée 4-fold sur votre sujet. Utiliser
un taux d’apprentissage réduit (10⁻⁴) et moins d’époques (50–100) pour éviter le surapprentissage. Cela
améliore typiquement la précision de 5 à 15% par rapport au modèle cross-sujet appliqué directement
[14][20].


## VI. Illettrisme BCI : un défi connu

On estime que 15 à 30% de la population générale ne peut pas contrôler efficacement un système MI-BCI,
même après entraînement [21]. Ce phénomène, appelé illettrisme BCI ou inefficience BCI, représente l’un
des défis les plus significatifs du domaine.

Ahn et al. [22] ont analysé les données EEG de 52 sujets et ont constaté que les individus BCI-illettrés
montrent des patterns spectraux caractéristiques même au repos : une puissance thêta (4–8 Hz) plus élevée
dans les régions frontales et une puissance alpha (8–13 Hz) plus faible globalement. Ces patterns persistent
à travers les états mentaux (repos, pré-IM, IM), suggérant qu’ils reflètent des traits individuels stables.

Implication pratique pour votre projet : Avant d’investir du temps dans l’entraînement IM, réalisez un
dépistage rapide pour chaque membre de l’équipe : enregistrez 2 minutes d’EEG au repos (yeux ouverts) sur
C3 et C4, calculez le spectre de puissance, et vérifiez la présence d’un pic mu clair autour de 10–12 Hz. Si le
pic mu est absent ou très faible, cette personne est potentiellement dans le groupe BCI-inefficient. Ce n’est
pas un échec — c’est un phénomène neurophysiologique documenté.

Blankertz et al. [21] ont proposé l’apprentissage co-adaptatif comme solution partielle : le classifieur s’adapte
à l’utilisateur tandis que l’utilisateur reçoit du feedback et apprend à moduler son activité cérébrale.
Commencer avec un ensemble de caractéristiques simple et passer progressivement à des caractéristiques
optimisées peut aider les utilisateurs précédemment inefficients à gagner le contrôle en une seule session.

## VII. Bonnes pratiques de prétraitement

Un prétraitement robuste est essentiel pour l’IM-BCI, surtout avec du matériel grand public où la qualité du
signal est inférieure aux systèmes cliniques.

1. Filtrage passe-bande : Appliquer un filtre Butterworth 4–40 Hz (ordre 4, phase nulle via filtfilt). La borne
inférieure à 4 Hz supprime les dérives lentes et les artefacts de mouvement. La borne supérieure à 40 Hz
supprime les artefacts musculaires (EMG) et le bruit secteur (50 Hz en Europe).
2. Filtre notch : Appliquer un filtre coupe-bande à 50 Hz pour supprimer l’interférence du réseau électrique.
L’OpenBCI est particulièrement sensible à cet artefact.
**3. Rejet d’artefacts :** Rejeter les époques avec une amplitude crête-à-crête dépassant 100 μV. Pour les
enregistrements plus longs, envisager l’utilisation de l’Artifact Subspace Reconstruction (ASR), une méthode
automatique qui identifie et corrige les segments de grande amplitude.
4. Filtrage spatial (Laplacien) : Pour les électrodes ayant des voisines disponibles (C3 entourée de FC3 et
CP3), calculer le Laplacien de surface : C3_lap = C3 - moyenne(FC3, CP3). Cela aiguise la résolution spatiale
et améliore significativement le rapport signal/bruit du rythme mu. C’est l’un des avantages principaux du
montage 8 canaux proposé par rapport à un montage à couverture large.
5. Extraction des époques : Découper les essais de 0,5 s à 4,0 s après l’apparition de l’indice. Le décalage
de 0,5 s évite le potentiel évoqué visuel du stimulus. La fenêtre de 3,5 s fournit suffisamment de données
pour une estimation fréquentielle fiable (résolution FFT = 1/3,5 ≈ 0,29 Hz).
6. Normalisation de la baseline : Soustraire la moyenne de la baseline pré-indice (1 seconde) de chaque
époque. Puis normaliser à variance unitaire par canal. Cela compense les variations d’amplitude inter-essais
et inter-sessions.


## VIII. Perspectives et améliorations

### A. Court terme (dans le cadre de ce projet)

1. BCI hybride : Combiner l’IM avec la détection de clignements oculaires pour un ensemble de commandes
plus riche. L’IM fournit le contrôle directionnel (gauche/droite), tandis que les doubles clignements
volontaires servent de commandes discrètes (sélectionner/confirmer). Aucun matériel supplémentaire n’est
nécessaire — les artefacts de clignement sont facilement détectables sur FC3/FC4.
2. Boucle de feedback en ligne : Implémenter la classification en temps réel avec feedback visuel (ex : un
curseur se déplaçant proportionnellement à la confiance du classifieur). Le feedback est crucial pour
l’apprentissage MI — les utilisateurs doivent savoir quand leur imagerie est efficace [2].
3. Classifieur adaptatif : Mettre à jour le classifieur de manière incrémentale pendant l’utilisation avec une
adaptation non supervisée (ex : moyenne mobile exponentielle des moyennes de classe). Cela compense la
non-stationnarité des signaux EEG au sein et entre les sessions.

### B. Moyen terme (itérations futures du projet)

1. Extension multi-classes : Ajouter l’imagerie des pieds comme troisième classe en utilisant l’électrode Cz.
Cela permet un contrôle 3 directions (gauche/droite/avant) ou un mécanisme de sélection.
2. Application de contrôle robotique : Interfacer la sortie du classifieur avec une main ou un bras robotique
via communication série. Comme montré sur l’image de description du projet, OpenBCI a déjà été utilisé
pour contrôler des prothèses robotiques. La sortie de classification (gauche/droite/repos) se traduit en
commandes motrices via une machine à états simple avec un seuil de confiance.
3. Augmentation de données : Appliquer des techniques d’augmentation spécifiques à l’EEG : segmentation
par fenêtre glissante, injection de bruit (gaussien additif), déformation temporelle (time warping) et dropout
de canaux. Ces techniques peuvent doubler ou tripler la taille effective du jeu d’entraînement [24].

### C. Long terme (perspectives de recherche)

1. Modèles indépendants du sujet : Les approches de pointe comme TFTL [25] utilisent l’adaptation de
domaine et l’alignement sur données de repos pour construire des modèles fonctionnant entre sujets et même
entre jeux de données sans aucune donnée de calibration labellisée de l’utilisateur cible.
2. Architectures basées sur les Transformers : Les mécanismes d’attention dans des modèles comme
CTNet [17] capturent des dépendances temporelles à longue portée que les CNN manquent. Ces architectures
sont prometteuses mais nécessitent plus de données et de calcul.
3. Intégration multimodale : Combiner l’EEG avec l’EMG (électromyographie), l’EOG (électro-
oculographie) ou la fNIRS (spectroscopie proche infrarouge fonctionnelle) peut améliorer la robustesse de
classification et fournir des canaux de contrôle complémentaires.

## IX. Conclusion

Ce rapport a présenté un cadre de recherche complet pour la construction d’une Interface Cerveau-Machine
basée sur l’Imagerie Motrice EEG. Les recommandations clés sont : (1) utiliser un montage sensorimoteur
focalisé à 8 canaux (C3, Cz, C4, FC3, FC4, CP3, CP4, FCz) ; (2) pré-entraîner sur PhysioNet EEGMMIDB
avec extraction de canaux correspondants ; (3) commencer par CSP+LDA comme baseline interprétable, puis
passer à EEGNet ; (4) utiliser l’Euclidean Alignment et le fine-tuning pour combler le décalage de domaine
PhysioNet–OpenBCI ; et (5) dépister l’illettrisme BCI précocement via l’évaluation de la puissance mu au


repos. Le pipeline de transfer learning proposé en section V fournit un chemin concret et implémentable, des
données publiques au contrôle en temps réel avec du matériel grand public.


## Références

[1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller et T. M. Vaughan, «Brain-computer interfaces for
communication and control,» Clinical Neurophysiology, vol. 113, no. 6, pp. 767–791, 2002. doi:
10.1016/S1388-2457(02)00057- 3

[2] G. Pfurtscheller et C. Neuper, «Motor imagery and direct brain-computer communication,» Proceedings of the
IEEE, vol. 89, no. 7, pp. 1123–1134, 2001. doi: 10.1109/5.

[3] G. Pfurtscheller et F. H. Lopes da Silva, «Event-related EEG/MEG synchronization and desynchronization: basic
principles,» Clinical Neurophysiology, vol. 110, no. 11, pp. 1842–1857, 1999. doi: 10.1016/S1388-
2457(99)00141- 8

[4] G. Pfurtscheller et A. Aranibar, «Event-related cortical desynchronization detected by power measurements of
scalp EEG,» Electroenceph. and Clinical Neurophysiology, vol. 42, no. 6, pp. 817–826, 1977.
[5] G. Pfurtscheller, C. Brunner, A. Schlögl et F. H. Lopes da Silva, «Mu rhythm (de)synchronization and EEG
single-trial classification of different motor imagery tasks,» NeuroImage, vol. 31, no. 1, pp. 153–159, 2006. doi:
10.1016/j.neuroimage.2005.12.
[6] G. Pfurtscheller, C. Neuper, C. Guger et al., «Current trends in Graz brain-computer interface (BCI) research,»
IEEE Trans. Rehabilitation Engineering, vol. 8, no. 2, pp. 216–219, 2000.
[7] G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer et J. R. Wolpaw, «BCI2000: A general-purpose brain-
computer interface (BCI) system,» IEEE Trans. Biomedical Engineering, vol. 51, no. 6, pp. 1034–1043, 2004.
doi: 10.1109/TBME.2004.827072. Dataset: https://physionet.org/content/eegmmidb/1.0.0/

[8] A. L. Goldberger et al., «PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for
complex physiologic signals,» Circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[9] Z. Shuqfa, A. Lakas et A. N. Belkacem, «Increasing accessibility to a large brain-computer interface dataset:
Curation of Physionet EEG Motor Movement/Imagery Dataset,» Data in Brief, vol. 53, p. 110181, 2024. doi:
10.1016/j.dib.2024.

[10] M. Tangermann et al., «Review of the BCI Competition IV,» Frontiers in Neuroscience, vol. 6, p. 55, 2012. doi:
10.3389/fnins.2012.00055. Données : https://www.bbci.de/competition/iv/

[11] H. Ramoser, J. Müller-Gerking et G. Pfurtscheller, «Optimal spatial filtering of single trial EEG during imagined
hand movement,» IEEE Trans. Rehabilitation Engineering, vol. 8, no. 4, pp. 441–446, 2000.

[12] K. K. Ang, Z. Y. Chin, H. Zhang et C. Guan, «Filter bank common spatial pattern algorithm on BCI Competition
IV Datasets 2a and 2b,» Frontiers in Neuroscience, vol. 6, p. 39, 2012. doi: 10.3389/fnins.2012.

[13] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung et B. J. Lance, «EEGNet: a compact
convolutional neural network for EEG-based brain-computer interfaces,» Journal of Neural Engineering, vol.
15, no. 5, p. 056013, 2018. doi: 10.1088/1741-2552/aace8c. Code: https://github.com/vlawhern/arl-eegmodels
[14] X. Wang, M. Hersche et al., «An accurate EEGNet-based motor-imagery brain-computer interface for low-power
edge computing,» arXiv:2004.00077, 2020.

[15] R. T. Schirrmeister et al., «Deep learning with convolutional neural networks for EEG decoding and
visualization,» Human Brain Mapping, vol. 38, no. 11, pp. 5391–5420, 2017. doi: 10.1002/hbm.
[16] H. Altaheri et al., «Physics-informed attention temporal convolutional network for EEG-based motor imagery
classification,» IEEE Trans. Industrial Informatics, vol. 19, no. 2, pp. 2249–2258, 2023.

[17] Y. Zhao et al., «CTNet: a convolutional transformer network for EEG-based motor imagery classification,»
Scientific Reports, vol. 14, p. 20237, 2024. doi: 10.1038/s41598- 024 - 71118 - 7

[18] AMEEGNet : «Attention-based multiscale EEGNet for effective motor imagery EEG decoding,» Frontiers in
Neurorobotics, vol. 19, 2025. doi: 10.3389/fnbot.2025.

[19] CIACNet : «A composite improved attention convolutional network for motor imagery EEG classification,»
PMC, 2025.

[20] D. Wu, Y. Xu et B.-L. Lu, «Transfer learning for EEG-based brain-computer interfaces: A review of progress
made since 2016,» IEEE Trans. Cognitive and Developmental Systems, vol. 14, no. 1, pp. 4–19, 2022. doi:
10.1109/TCDS.2020.
[21] C. Vidaurre et B. Blankertz, «Towards a cure for BCI illiteracy,» Brain Topography, vol. 23, pp. 194–198, 2010.
doi: 10.1007/s10548- 009 - 0121 - 6


[22] M. Ahn, H. Cho, S. Ahn et S. C. Jun, «High theta and low alpha powers may be indicative of BCI-illiteracy in
motor imagery,» PLoS ONE, vol. 8, no. 11, p. e80886, 2013. doi: 10.1371/journal.pone.
[23] H. He et D. Wu, «Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach,»
IEEE Trans. Biomedical Engineering, vol. 67, no. 2, pp. 399–410, 2020. doi: 10.1109/TBME.2019.

[24] F. Lotte, «Signal processing approaches to minimize or suppress calibration time in oscillatory activity-based
brain-computer interfaces,» Proceedings of the IEEE, vol. 103, no. 6, pp. 871–890, 2015.
[25] TFTL : «A task-free transfer learning strategy for EEG-based cross-subject & cross-dataset motor imagery BCI,»

2024. Utilise ShallowConvNet, EEGNet et TCNet-Fusion comme baselines sur 5 datasets incluant PhysioNet
MI.



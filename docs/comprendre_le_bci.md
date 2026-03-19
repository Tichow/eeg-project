# Comprendre le BCI — Du signal cerebral au controle par la pensee

Ce document explique comment fonctionne notre interface cerveau-ordinateur (BCI), de la neurophysiologie sous-jacente jusqu'au controle temps reel.

---

## 1. Pourquoi les electrodes captent quelque chose

### L'activite electrique du cerveau

Le cerveau contient ~86 milliards de neurones. Quand un groupe de neurones s'active en meme temps, leurs courants post-synaptiques s'additionnent et creent un champ electrique mesurable a la surface du crane. C'est ce que l'EEG (electroencephalographie) capte : des differences de potentiel de l'ordre de quelques **microvolts** (µV).

Un seul neurone est invisible pour l'EEG. Il faut l'activation synchrone de **dizaines de milliers de neurones** orientes dans la meme direction pour produire un signal detectable. C'est pour cela que l'EEG a une bonne resolution temporelle (milliseconde) mais une mauvaise resolution spatiale (centimetres) — il capte l'activite de larges populations neuronales.

### Rythmes cerebraux et bandes de frequence

L'activite EEG n'est pas aleatoire. Les neurones s'organisent en oscillations rythmiques a differentes frequences :

| Bande | Frequence | Etat associe |
|-------|-----------|--------------|
| **Delta** | 0.5–4 Hz | Sommeil profond |
| **Theta** | 4–8 Hz | Somnolence, memoire |
| **Alpha (mu)** | 8–13 Hz | Repos, yeux fermes, inactivite motrice |
| **Beta** | 13–30 Hz | Concentration, activite motrice |
| **Gamma** | 30+ Hz | Traitement cognitif intense |

Pour l'imagerie motrice, deux bandes sont cruciales : **mu (8-13 Hz)** et **beta (13-30 Hz)**.

### ERD/ERS : la cle de l'imagerie motrice

Quand tu **imagines** bouger ta main droite, les neurones du cortex moteur gauche (hemisphere contralateral) se desynchronisent : leur rythme mu/beta diminue en puissance. C'est l'**Event-Related Desynchronization (ERD)** — une baisse d'amplitude dans la bande 8-30 Hz.

Quand tu arrete d'imaginer, le rythme revient avec un rebond : c'est l'**Event-Related Synchronization (ERS)**.

Le point crucial : l'ERD est **lateralise**. Imaginer la main droite produit un ERD surtout sur **C3** (hemisphere gauche). Imaginer la main gauche produit un ERD sur **C4** (hemisphere droit). C'est cette asymetrie que le classifieur exploite.

Pour **mains vs pieds**, la separation est encore plus nette : les mains activent la zone laterale du cortex moteur (C3/C4), les pieds activent la zone mediane (Cz). Ces zones sont anatomiquement plus eloignees, ce qui explique pourquoi cette tache est plus facile a classifier (~71% vs ~63%).

---

## 2. Placement des electrodes

### Pourquoi ces 8 canaux

Notre casque OpenBCI Cyton a 8 canaux. On les place strategiquement autour du cortex moteur :

```
          FC1    FC2          ← premoteur (planification du mouvement)
       C3    Cz    C4        ← moteur primaire (execution/imagerie)
             Pz              ← parietal (integration sensorimotrice)
          O1    O2           ← occipital (reference visuelle)
```

| Electrode | Pourquoi elle est la |
|-----------|---------------------|
| **C3** | Cortex moteur droit — ERD quand on imagine la main droite |
| **C4** | Cortex moteur gauche — ERD quand on imagine la main gauche |
| **Cz** | Cortex moteur midline — ERD quand on imagine les pieds |
| **FC1/FC2** | Premoteur — activite de planification motrice |
| **Pz** | Parietal — integration sensorimotrice, donne du contexte spatial |
| **O1/O2** | Occipital — capte le rythme alpha visuel, aide a la stabilite du casque |

### L'electrode de reference (SRB2)

L'EEG mesure des **differences de potentiel** entre chaque electrode et une reference. Sur le Cyton, la reference est le pin **SRB2**, ou on branche un clip sur le lobe de l'oreille (earlobe). Sans cette reference, tous les canaux saturent (railage).

---

## 3. Preprocessing : nettoyer le signal

Le signal brut du Cyton est bruiete. Avant de classifier, on le nettoie :

### Bandpass 8-30 Hz

On garde uniquement les frequences mu (8-13 Hz) et beta (13-30 Hz) — la ou se trouve l'information motrice. Tout ce qui est en dessous (drift des electrodes, mouvements des yeux) et au-dessus (bruit musculaire, bruit electronique) est elimine.

Implementation : filtre Butterworth 4eme ordre, applique de facon zero-phase (`sosfiltfilt`) pour ne pas decaler les signaux dans le temps.

### Notch 50 Hz

Le reseau electrique europeen oscille a 50 Hz et contamine le signal EEG. Un filtre notch (coupe-bande) elimine cette frequence specifique sans affecter le reste.

### Rejet d'artefacts

Les epochs ou l'amplitude depasse **500 µV** (pic a pic) sont supprimees. Un signal EEG normal depasse rarement 100-200 µV. Au-dela de 500 µV, c'est un artefact (mouvement de machoire, clignement, mauvais contact electrode).

---

## 4. Classification : CSP + LDA

### Etape 1 — Decoupage en epochs

Le signal continu est decoupe autour de chaque annotation T1/T2 :

```
Annotation T1 (onset)
    |
    v
----[0.5s ————— 3.5s]----->  = epoch de 3 secondes
     |                  |
   tmin              tmax
```

On prend de 0.5s a 3.5s apres l'onset du cue. Les 0.5 premieres secondes sont ignorees car le sujet n'a pas encore commence a imaginer (temps de reaction). On obtient un tableau 3D : `(n_epochs, 8 canaux, 750 echantillons)`.

### Etape 2 — CSP (Common Spatial Patterns)

C'est le coeur du systeme. Le CSP est un algorithme d'extraction de features specifiquement concu pour les problemes a 2 classes en EEG.

**Le probleme** : on a 8 canaux, mais l'information discriminante est diluee dans le melange de tous les signaux. L'ERD sur C3 n'est pas "propre" — il est contamine par l'activite des electrodes voisines (probleme de conduction volumique).

**L'idee du CSP** : trouver des combinaisons lineaires des 8 canaux (des "filtres spatiaux") qui **maximisent la variance** pour une classe et la **minimisent** pour l'autre.

Concretement, le CSP apprend une matrice W de taille (8 × 8). En multipliant l'epoch par W, on obtient 8 "canaux virtuels" :
- Les premiers canaux virtuels ont une variance maximale pour la classe 1 (ex: main gauche) et minimale pour la classe 2
- Les derniers canaux virtuels font l'inverse

On garde les **6 composantes les plus discriminantes** (3 premieres + 3 dernieres). Pour chaque epoch, on calcule la **log-variance** de ces 6 canaux virtuels, ce qui donne un vecteur de 6 features.

**Regularisation Ledoit-Wolf** : avec seulement 8 canaux et ~90 epochs, les matrices de covariance par classe sont instables. La regularisation Ledoit-Wolf "retrecit" (shrink) la covariance estimee vers une matrice identite, ce qui stabilise le CSP et evite le surapprentissage.

### Etape 3 — LDA (Linear Discriminant Analysis)

Le LDA recoit les 6 features CSP de chaque epoch et trace un hyperplan qui separe les 2 classes. C'est un classifieur lineaire simple et rapide, bien adapte aux features CSP car :
- Peu de features (6) → pas besoin d'un modele complexe
- Les features CSP sont approximativement gaussiennes (log-variance)
- Le LDA fournit des **probabilites** via `predict_proba()`, utiles pour le temps reel

### Etape 4 — Cross-validation

On evalue le modele avec une **cross-validation stratifiee 5-fold** :
1. Les epochs sont divisees en 5 groupes de taille egale (meme ratio T1/T2 dans chaque groupe)
2. On entraine sur 4 groupes, on teste sur le 5eme
3. On repete 5 fois en changeant le groupe test
4. L'accuracy finale est la moyenne des 5 folds

Cela donne une estimation honnete de la performance : le modele est toujours teste sur des donnees qu'il n'a pas vues pendant l'entrainement.

### Visualisation du pipeline complet

```
 Signal brut (8 ch × N samples)
       |
       v
 [Bandpass 8-30 Hz + Notch 50 Hz]
       |
       v
 Epochs (n_epochs × 8 ch × 750 samples)
       |
       v
 [Rejet artefacts > 500 µV]
       |
       v
 [CSP : 8 canaux → 6 features (log-variance)]
       |
       v
 [LDA : 6 features → classe 0 ou 1 + probabilites]
       |
       v
 Label (T1 ou T2) + P(T1), P(T2)
```

---

## 5. Pourquoi ca marche (et pourquoi ca marche pas toujours)

### Pourquoi ca marche

L'imagerie motrice active les **memes zones corticales** que le mouvement reel, mais avec une amplitude plus faible. Le CSP est capable de detecter cette difference d'amplitude (variance) entre les classes, meme quand elle est subtile. Avec 3 runs de 15 essais par classe (= 90 epochs), il a assez de donnees pour estimer des filtres spatiaux fiables.

### Variabilite inter-sujets

Sur les 109 sujets PhysioNet :
- ~10% depassent 90% de precision
- ~25% restent au niveau du hasard (~50%)
- La majorite est entre 55-75%

Cette variabilite est un phenomene connu en BCI appele **"BCI illiteracy"** : certaines personnes produisent des patterns moteurs tres nets, d'autres non. Ca depend de l'anatomie corticale, de la capacite a produire de l'imagerie kinesthesique, et de la concentration.

### Mouvement reel vs imagerie

Le mouvement reel donne toujours de meilleurs resultats car l'ERD est plus fort (les neurones moteurs s'activent vraiment, pas seulement "imaginent"). C'est pour ca que la Tache 3 (mouvement) sert de validation : si le casque capte bien le mouvement reel, on sait que le setup est correct.

---

## 6. Controle temps reel

### Du offline au online

L'entrainement se fait **offline** : on enregistre des donnees, on les analyse apres coup. Le controle se fait **online** : le modele predit en continu pendant que le sujet pense.

### Architecture temps reel

```
 Casque Cyton (250 Hz)
       |
       | chunks de ~25 samples toutes les 100ms
       v
 [Buffer circulaire : 3 secondes = 750 samples]
       |
       | toutes les 0.5 secondes
       v
 [Filtrage causal : bandpass 8-30 Hz + notch 50 Hz]
       |
       v
 [Pipeline CSP+LDA : predict + predict_proba]
       |
       v
 (label, probabilites) → signal Qt → UI
       |
       v
 [Lissage EMA] → mise a jour balle + barres
```

### Buffer circulaire (ring buffer)

Le signal arrive en continu par morceaux de ~25 echantillons (100ms × 250 Hz). On les accumule dans un buffer circulaire de 750 echantillons (3 secondes). Quand le buffer est plein, les nouvelles donnees ecrasent les plus anciennes — on a toujours les 3 dernieres secondes.

### Fenetre glissante

Toutes les 0.5 secondes (125 echantillons), on extrait le contenu du buffer, on le filtre, et on le passe au modele. Cela donne **~2 predictions par seconde** avec un chevauchement de 83% entre fenetres consecutives (2.5s en commun sur 3s).

### Filtrage causal vs zero-phase

L'entrainement utilise `sosfiltfilt` (filtre zero-phase, non-causal) : il lit le signal dans les deux sens, ce qui elimine le dephasage. C'est impossible en temps reel car on ne connait pas le futur.

On utilise donc `sosfilt` (filtre causal) : il lit le signal dans un seul sens. Cela introduit un leger dephasage, mais les ~50 premieres millisecondes de transitoire (warm-up du filtre) sur une fenetre de 3 secondes sont negligeables. Le CSP apprend des **patterns spatiaux** (differences de variance entre canaux), pas temporels — le dephasage n'affecte pas la variance.

### Lissage EMA (Exponential Moving Average)

Les predictions brutes fluctuent d'une fenetre a l'autre. Pour stabiliser l'affichage, on applique un lissage exponentiel :

```
smoothed = alpha × prediction_actuelle + (1 - alpha) × smoothed_precedent
```

- **alpha = 1.0** : pas de lissage, tres reactif mais instable
- **alpha = 0.3** (defaut) : bon compromis reactivite/stabilite
- **alpha = 0.1** : tres lisse, lent a reagir

### Seuil de confiance

Le LDA retourne des probabilites (ex: P(mains)=0.72, P(pieds)=0.28). On ne declenche une "commande" que si la probabilite lissee depasse le **seuil de confiance** (defaut 60%). En dessous, la balle reste au centre — le systeme dit "je ne suis pas sur".

Augmenter le seuil (ex: 0.75) reduit les faux positifs mais rend le controle plus exigeant. Le baisser (ex: 0.55) rend le systeme plus reactif mais moins fiable.

---

## 7. Limites et perspectives

### Limites actuelles

- **2 classes seulement** : le CSP binaire ne gere que 2 classes. Pour plus de commandes, il faudrait un CSP multi-classe (one-vs-rest) ou d'autres approches
- **Latence ~3 secondes** : il faut remplir le buffer avant la premiere prediction, et chaque prediction porte sur les 3 dernieres secondes — pas instantane
- **Fatigue mentale** : l'imagerie motrice demande de la concentration. Apres 10-15 minutes, les performances degradent
- **Variabilite inter-sessions** : un modele entraine un jour peut etre moins bon le lendemain (position des electrodes, etat mental, impedance)

### Pistes d'amelioration

- **Adaptive CSP** : re-calibrer le modele pendant l'utilisation
- **Feedback auditif** : ajouter un son quand une commande est detectee
- **Deep learning** : remplacer CSP+LDA par un reseau convolutif (EEGNet) pour capturer des patterns temporels
- **Plus de commandes** : combiner plusieurs taches (gauche/droite + mains/pieds) pour 4 commandes

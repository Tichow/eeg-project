# Protocole — Effet Berger (yeux ouverts / yeux fermés)

Guide pas-à-pas pour enregistrer un test EEG propre et observer le pic alpha (~10 Hz) à la fermeture des yeux. Ce protocole est conçu pour le casque **OpenBCI Ultracortex Mark IV** (Cyton 8 canaux, 250 Hz, électrodes sèches ThinkPulse).

---

## 1. Matériel nécessaire

- Casque OpenBCI Ultracortex Mark IV monté et connecté (Cyton + dongle USB)
- Ordinateur avec le projet `fabien-eeg` installé (cf. `pip install -r requirements.txt`)
- Pièce calme, éclairage constant (pas de néons)
- Chaise confortable pour le sujet
- Un point de fixation sur le mur en face (post-it, autocollant) à hauteur des yeux, à ~2 m

---

## 2. Préparation du sujet (5 min)

### 2.1 Le sujet

- Cheveux propres et **secs** (pas de gel/cire — augmente l'impédance)
- Pas de caféine depuis 2 h (modifie le rythme alpha)
- Retirer boucles d'oreilles, piercings métalliques au niveau de la tête
- S'asseoir confortablement, dos droit, pieds à plat, **mains posées sur les cuisses** (pas sur la table)

### 2.2 Mise en place du casque

1. Placer le casque sur la tête du sujet. Le repère est : **Cz** (sommet du crâne) doit être à mi-chemin entre le nasion (racine du nez) et l'inion (bosse à l'arrière du crâne).

2. **Régler les pattes** pour que le casque soit bien serré sans être douloureux. Il ne doit pas bouger quand le sujet tourne légèrement la tête.

3. **Pour chaque électrode**, surtout O1 (CH7) et O2 (CH8) :
   - Écarter les cheveux sous les picots avec les doigts ou un stylo
   - Les picots ThinkPulse doivent toucher **directement le cuir chevelu**
   - Appuyer doucement l'électrode vers le bas et relâcher — elle doit rester en contact
   - Si les cheveux sont épais, prendre le temps de bien dégager : c'est l'étape la plus importante

4. **Vérifier le clip d'oreille** (référence SRB) : il doit être bien clipsé sur le lobe d'oreille, avec un bon contact peau. Essayer les deux oreilles et garder celle où le signal est le plus stable.

### 2.3 Environnement

- **Éteindre** tout écran non utilisé (moniteur secondaire, TV)
- **Éloigner** les téléphones portables (> 1 m du sujet)
- **Débrancher** les chargeurs proches du sujet
- **Éteindre** ventilateur/climatisation (vibrations = artefacts)
- L'**opérateur** s'assoit derrière ou à côté du sujet, pas en face (le sujet ne doit pas être distrait)
- Fenêtres fermées si rue bruyante

---

## 3. Lancement et vérification du signal (5 min)

### 3.1 Démarrer le dashboard temps réel

Dans le terminal, depuis le dossier du projet :

```bash
python main.py
```

Choisir l'option :

```
2
```

(Temps réel — OpenBCI Cyton)

Quand il demande le port série, appuyer Entrée pour le port par défaut, ou taper le port si différent :

```
/dev/cu.usbserial-DM03H2DU
```

Canaux à afficher → appuyer Entrée (tous les canaux).

Le dashboard s'ouvre avec 4 panels : signal temporel, PSD, SNR, spectrogramme.

### 3.2 Activer le traitement du signal

Dans la sidebar droite, section **Traitement**, vérifier que :

- [x] **Passe-bande (1–50 Hz)** → coché
- [x] **Notch 50 Hz** → coché
- [x] **CAR** → **cocher** (désactivé par défaut, mais essentiel pour réduire le bruit 50 Hz commun)
- [ ] SNR Alpha → optionnel, peut le cocher si on veut voir le SNR en direct

### 3.3 Vérifier la qualité du signal

**Sur le panel du signal temporel (en haut)**, pour chaque canal, vérifier :

| Ce qu'on voit | Diagnostic | Action |
|---|---|---|
| Signal entre ±30 et ±80 µV, oscillations variées | **BON** — signal EEG normal | Continuer |
| Signal > ±200 µV, pics très amples | **Mauvais contact** | Réajuster l'électrode, écarter les cheveux |
| Signal plat (quasiment une ligne droite) | **Pas de contact** ou saturation | Vérifier que l'électrode touche le cuir chevelu |
| Oscillation très régulière et rapide (50 Hz pur) | **Bruit secteur dominant** | Vérifier le clip d'oreille, éloigner les sources électriques |
| Signal qui dérive lentement (monte puis descend) | **Dérive DC** — normal avec les électrodes sèches | Pas grave si l'amplitude reste < 200 µV |

**Les canaux les plus importants pour ce test sont CH7 (O1) et CH8 (O2).** Si ceux-là sont mauvais, inutile de continuer — réajuster les électrodes occipitales.

### 3.4 Test rapide de l'alpha

Demander au sujet de **fermer les yeux 10 secondes** puis de les **rouvrir**.

Sur le **panel PSD** (2e panel), observer les courbes de CH7 et CH8 :
- Si on voit la puissance augmenter autour de 8–13 Hz quand les yeux se ferment → le signal est bon, on peut enregistrer
- Si aucun changement visible → réajuster O1/O2, vérifier le contact

---

## 4. Enregistrement (5 min)

### 4.1 Démarrer l'enregistrement

1. Appuyer sur la touche **R** dans la fenêtre du dashboard

2. La boîte de dialogue "Nouvel enregistrement" apparaît :
   - **Sujet** : taper le prénom du sujet (ex : `Mattéo`, `Fabien`, `Némo`)
   - **Type de test** : sélectionner **Yeux fermés**
   - Cliquer **OK**

3. Le titre de la fenêtre affiche `[● REC 00:00]` — l'enregistrement est en cours.

### 4.2 Protocole d'enregistrement

L'opérateur guide le sujet **à voix haute** et pose les marqueurs avec la touche **M** :

```
PHASE 1 — YEUX OUVERTS (60 secondes)
──────────────────────────────────────
Opérateur dit : "On commence. Fixe le point sur le mur. Ne bouge pas."
Opérateur appuie M puis sélectionnera "Ouvrir les yeux" à la fin.
Attendre 60 secondes (regarder le chrono [● REC] dans le titre).

PHASE 2 — YEUX FERMÉS (60 secondes)
──────────────────────────────────────
Opérateur dit : "Ferme les yeux. Reste détendu. Ne serre pas les paupières."
Opérateur appuie M (marqueur).
Attendre 60 secondes.

PHASE 3 — YEUX OUVERTS (60 secondes)
──────────────────────────────────────
Opérateur dit : "Ouvre les yeux. Fixe le point."
Opérateur appuie M (marqueur).
Attendre 60 secondes.

PHASE 4 — YEUX FERMÉS (60 secondes)
──────────────────────────────────────
Opérateur dit : "Ferme les yeux."
Opérateur appuie M (marqueur).
Attendre 60 secondes.

FIN
──────────────────────────────────────
Opérateur dit : "C'est terminé, tu peux te détendre."
```

**Durée totale : 4 minutes.**

### 4.3 Consignes au sujet pendant l'enregistrement

Dire au sujet **avant** de commencer :

> "Pendant tout le test, tu dois rester **complètement immobile**. Ne bouge pas la tête, ne serre pas la mâchoire, ne te gratte pas. Quand tes yeux sont ouverts, fixe le point sur le mur sans balayer du regard. Quand tes yeux sont fermés, ne serre pas les paupières, ferme-les normalement et ne pense à rien de particulier. Si tu as besoin de bouger ou de déglutir, fais-le pendant que je te parle entre les phases, pas pendant le silence."

### 4.4 Arrêter l'enregistrement

1. Appuyer sur **R** pour arrêter.

2. La boîte de dialogue de fin apparaît avec la liste des marqueurs posés.

3. **Labelliser chaque marqueur** dans l'ordre :
   - Marqueur 1 → `Ouvrir les yeux`
   - Marqueur 2 → `Fermer les yeux`
   - Marqueur 3 → `Ouvrir les yeux`
   - Marqueur 4 → `Fermer les yeux`

4. Dans **Notes**, écrire : `Protocole Berger 4x60s. Commence yeux ouverts.`

5. Cliquer **OK**. Le fichier est sauvegardé dans `recordings/`.

---

## 5. Analyse des résultats

### 5.1 Ouvrir l'enregistrement

Dans le terminal :

```bash
python main.py
```

Choisir l'option :

```
4
```

(Analyser un enregistrement)

Le browser s'ouvre. Filtrer par :
- **Sujet** : le sujet testé
- **Test** : Yeux fermés

Sélectionner l'enregistrement (le plus récent est en haut) et cliquer **Analyser**.

### 5.2 Ce qui s'affiche

La fenêtre **PSD par section** s'ouvre avec une courbe par segment :
- Courbes **bleues** = segments "Yeux fermés"
- Courbes **rouges** = segments "Yeux ouverts"

En haut à gauche, un sélecteur **Canal** permet de choisir quel canal afficher.

### 5.3 Comment analyser

**Étape 1 — Sélectionner le bon canal :**

Changer le sélecteur de canal :
- **CH7** = O1 (occipital gauche) — le plus important
- **CH8** = O2 (occipital droit) — le second plus important

Ne pas rester sur "Moyenne" — la moyenne de tous les canaux dilue le pic alpha.

**Étape 2 — Observer la bande 8–13 Hz (Alpha) :**

La zone Alpha est mise en évidence en orange sur le graphique.

| Ce qu'on observe | Interprétation |
|---|---|
| Les courbes bleues (yeux fermés) ont un **pic net** entre 8 et 13 Hz, nettement au-dessus des courbes rouges (yeux ouverts) | **Effet Berger clair.** Le sujet produit un alpha occipital normal. Signal de bonne qualité. |
| Les courbes bleues sont légèrement au-dessus des rouges dans la bande alpha, mais pas de pic net | **Effet Berger faible.** Peut être dû à : contact O1/O2 insuffisant, sujet stressé/pas détendu, segments trop courts. Refaire le test avec un meilleur contact. |
| Aucune différence entre bleu et rouge | **Pas d'effet Berger visible.** Causes possibles : (1) mauvais contact O1/O2, (2) sujet alpha-naïf (rare, ~5%), (3) trop de bruit. Vérifier d'abord le contact. |
| Un pic très fin et pointu à exactement 50 Hz | **Bruit secteur résiduel.** Le notch ne suffit pas. Vérifier le clip d'oreille et l'environnement. |
| Puissance très élevée (> 10^4 µV²/Hz) partout | **Artefact majeur.** Probablement un mauvais contact ou mouvement pendant l'enregistrement. |

**Étape 3 — Comparer les deux segments de chaque condition :**

Vous avez 2 segments "Yeux fermés" (bleu clair et bleu foncé) et 2 segments "Yeux ouverts" (rouge clair et rouge foncé). Si les deux bleus se ressemblent entre eux et les deux rouges aussi → le signal est **reproductible**, c'est fiable.

Si les courbes d'une même condition sont très différentes → il y a eu du bruit ou un mouvement pendant un des segments.

### 5.4 Critères de réussite

Un bon enregistrement doit montrer, sur CH7 (O1) ou CH8 (O2) :

1. **Pic alpha visible** : une bosse nette entre 8 et 13 Hz sur les courbes yeux fermés
2. **Différence yeux fermés / yeux ouverts** : la puissance alpha doit être au moins **2 à 5 fois** plus élevée yeux fermés (0.3 à 0.7 décades de différence sur l'échelle log)
3. **Reproductibilité** : les deux segments yeux fermés doivent se ressembler
4. **Pas de 50 Hz dominant** : pas de pic fin à 50 Hz qui domine tout le spectre

### 5.5 Si les résultats ne sont pas bons

Checklist de diagnostic :

1. **Le pic est présent sur CH7 mais pas CH8 (ou l'inverse)** → L'électrode sans pic a un mauvais contact. Réajuster et refaire.

2. **Aucun pic sur aucun canal** → Tester aussi CH5 (P7) et CH6 (P8). Si des pics apparaissent là mais pas sur O1/O2, c'est un problème de contact occipital.

3. **Le spectre est dominé par le 50 Hz** → Le clip d'oreille a un mauvais contact, ou il y a une source de bruit électrique trop proche.

4. **Le sujet dit qu'il n'arrive pas à se détendre** → Faire une pause, recommencer. Le stress supprime l'alpha.

5. **Les courbes yeux fermés et yeux ouverts sont identiques** → Demander au sujet s'il a bien fermé les yeux pendant les bonnes phases. Vérifier les marqueurs.

---

## 6. Pour aller plus loin

### 6.1 Comparer avec PhysioNet

Pour vérifier que votre pipeline fonctionne, tester d'abord sur les données PhysioNet (qui sont propres et doivent montrer un effet Berger net) :

```bash
python main.py
```

Choisir **1** (Analyse offline PhysioNet), puis **S001**, run **2** (R02 — yeux fermés).

Sélectionner les canaux `O1 OZ O2` quand demandé. Vous devez voir un pic alpha clair sur la PSD.

### 6.2 Remarque importante sur le pipeline actuel

L'analyse via l'option 4 (`run_eyes_closed_analysis`) affiche la PSD sur les données **brutes** (sans filtrage bandpass/notch). Pour une analyse plus propre, utiliser l'option **5** (Dashboard offline) qui applique le filtrage en temps réel pendant la lecture, et activer bandpass + notch + CAR dans la sidebar.

---

## Annexe — Résumé en une page

```
PROTOCOLE BERGER — RÉSUMÉ RAPIDE
═══════════════════════════════════

PRÉPARER (5 min)
  • Cheveux dégagés sous O1/O2, casque bien serré
  • Clip d'oreille en contact, pièce calme
  • python main.py → choix 2 → port par défaut → Entrée

VÉRIFIER (2 min)
  • Sidebar : cocher Passe-bande + Notch + CAR
  • Vérifier CH7/CH8 : amplitude 30–80 µV, pas de saturation
  • Test rapide : yeux fermés 10 s → pic sur PSD ?

ENREGISTRER (4 min)
  • R → Sujet + "Yeux fermés" → OK
  • M + "Ouvre les yeux" → 60 s fixation point
  • M + "Ferme les yeux" → 60 s relâché
  • M + "Ouvre les yeux" → 60 s fixation point
  • M + "Ferme les yeux" → 60 s relâché
  • R → labelliser → Notes → OK

ANALYSER
  • python main.py → choix 4 → sélectionner l'enregistrement
  • Canal : CH7 (O1) ou CH8 (O2)
  • Pic alpha 8–13 Hz yeux fermés > yeux ouverts = SUCCÈS
```

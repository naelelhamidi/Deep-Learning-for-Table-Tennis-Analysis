# Deep-Learning-for-Table-Tennis-Analysis

## Contexte et objectif

Sur les vidéos de matchs professionnels, le tracking donne, image par image, la position 2D de la balle dans l'image. Il ne donne pas sa profondeur : un point de l'image correspond à toute une droite de positions 3D possibles. Seuls les rebonds sur la table lèvent cette ambiguïté, puisque la balle s'y trouve à une hauteur connue.

On cherche à estimer, pour chaque coup, deux positions 3D :

la position de la balle à la frappe (d'où le joueur joue) ;
la position de la balle au passage du plan du filet (avec quelle marge elle franchit le filet).

Le repère 3D a son origine au centre de la table, au niveau du sol ; l'axe x est le long de la table, donc le plan du filet est le plan x = 0.

But ultime : utiliser ces deux positions pour estimer, en temps réel, la probabilité de gagner le point.

Aucune mesure 3D n'existe sur les vrais matchs. L'idée est donc de s'appuyer sur la physique du vol de la balle : on simule un grand nombre de trajectoires dont la 3D est connue, on les projette dans le plan de la caméra, et on entraîne un réseau de neurones à faire le chemin inverse (2D → 3D).

## Démarche et avancement
*	Déterminer le plan de la caméra (comment un point 3D devient un point d'image)	
*	Simuler de nombreuses trajectoires 3D connues et les projeter en 2D	
*	Séquencer le tracking des vrais matchs pour qu'il ait la même forme que les trajectoires simulées	
*	Entraîner un réseau 2D → 3D, puis le tester sur de vrais matchs	
*	Estimer la probabilité de gagner le point à partir des positions 3D

## Simulation des trajectoires qui constituent la base d'entraînement du RNN
### Génération des trajectoires 3D

Les trajectoires sont obtenues en intégrant les équations du mouvement de la balle entre deux contacts (raquette, table), avec un schéma de Runge-Kutta d'ordre 4 à pas de temps fin. Trois forces sont prises en compte :

le poids ;
la traînée aérodynamique (frottement de l'air, quadratique en la vitesse) ;
la force de Magnus, due à la rotation de la balle (topspin, backspin, effets latéraux).

Chaque trajectoire part d'un état initial tiré au hasard : position de frappe dans la zone de jeu, vitesse, direction, rotation sur les trois axes. Les coefficients physiques varient légèrement d'une trajectoire à l'autre pour ne pas dépendre d'un jeu de valeurs précis. Seules les trajectoires réalistes sont conservées : elles passent au-dessus du filet et rebondissent dans le camp adverse.

Les positions 3D sont ensuite échantillonnées à la cadence de la vidéo, puis projetées avec le plan de caméra pour obtenir des trajectoires 2D de même nature que le tracking réel.

### Le plan de la caméra : projection des trajectoires en 2D

Le plan de la caméra relie le monde simulé (3D connue) et le monde réel (2D seulement). La caméra de diffusion est la même sur tous les matchs, et on fait abstraction des zooms et autres perturbations. On la détermine une seule fois à partir de repères de dimensions connues (coins de la table, filet), puis on vérifie que d'autres points connus, non utilisés pour la déterminer, se projettent au bon endroit.

Cette étape est critique : si la caméra virtuelle de la simulation diffère de la vraie, le réseau apprend une relation 2D → 3D erronée, avec un biais systématique que l'entraînement ne peut pas corriger. Comme la caméra est éloignée, la perspective est faible : la profondeur est la grandeur la moins bien contrainte par l'image et la plus sensible à une erreur de calibration.

### Validation (code de validation)

Un second code permet de valider le réalisme des trajectoires simulées : il les superpose à de vraies trajectoires issues du tracking, pour vérifier que la simulation (physique + projection) reproduit bien ce que l'on observe dans les vrais matchs.

On obtient finalement des milliers de trajectoires auxquelles on ajoute un bruit gaussien pour simulées les incertitudes des données issues du tracking de vrais matchs

## Le réseau de neurones
### Le problème

C'est une régression supervisée sur des séquences : en entrée, la suite des positions 2D (x_px, y_px) d'une trajectoire, de longueur variable ; en sortie, la position 3D (x_m, y_m, z_m) à chaque image. L'entraînement minimise l'erreur quadratique moyenne (MSE) entre la sortie du réseau et la vraie 3D de la simulation. L'erreur est mesurée en centimètres, axe par axe.

### Pourquoi un LSTM bidirectionnel
LSTM : un réseau « à mémoire » qui lit la séquence image par image et retient ce qui est utile. C'est indispensable ici : une position 2D isolée ne permet pas de retrouver la profondeur, c'est la forme de la trajectoire entière qui la révèle (la gravité donne l'échelle, les rebonds sur la table donnent une ancre).
Bidirectionnel : il lit la trajectoire dans les deux sens, donc la profondeur d'une image s'appuie à la fois sur le passé et sur le futur (par exemple le rebond qui suit). Comme on reconstruit les trajectoires après coup, c'est autorisé, et c'est de loin le plus gros gain mesuré.

### Entrainement et optimisation du réseau

**Découpage par trajectoire (70 % entraînement / 15 % validation / 15 % test)**	: une trajectoire n'est jamais coupée en deux, et le jeu de test reste réellement « jamais vu ».

**Normalisation des entrées et des sorties, calculée sur le jeu d'entraînement uniquement** :	Met x, y, z à la même échelle (sinon x, qui varie sur ±2,9 m, écraserait z) et évite toute fuite d'information vers le test. Les erreurs sont ensuite reconverties en cm.

**Longueurs variables : complétion par des zéros (padding) + packing** : Les trajectoires font de 12 à 83 images. Le packing empêche le LSTM de lire les zéros de remplissage, ce qui est crucial en bidirectionnel.

**Perte masquée** :	La MSE ignore les images de remplissage : seules les vraies images comptent dans l'erreur.

**Une sortie 3D par image** :	Le réseau reconstruit la trajectoire 3D complète, d'où l'on peut ensuite lire les positions utiles (frappe, filet).

**Optimiseur Adam** :	Règle le pas de la descente de gradient séparément pour chaque paramètre. Convergence rapide et peu sensible au réglage : bon choix par défaut pour un LSTM.

**Gradient clipping** :	Plafonne la taille du gradient à chaque mise à jour pour éviter qu'il n'explose (entraînement qui part en NaN).

**Portes du LSTM** : Ce sont elles, et non le clipping, qui limitent le problème inverse, le vanishing gradient (gradient qui s'évanouit sur les séquences longues). C'est la raison d'être d'un LSTM par rapport à un réseau récurrent simple.

**Arrêt anticipé (early stopping)** :	On surveille l'erreur sur le jeu de validation : l'entraînement s'arrête quand elle ne progresse plus, et on restaure le meilleur modèle. Protège contre le surapprentissage et économise du temps de calcul.

**Baselines** :	Deux références pour juger le réseau : prédire la moyenne, et un réseau sans mémoire qui traite chaque point isolément.

**Test de santé** :	Le réseau doit être capable de sur-apprendre un tout petit lot de trajectoires (erreur ≈ 0). Sinon, c'est qu'il y a un bug, et inutile de régler quoi que ce soit.

### Choix des hyperparamètres
Afin de trouver l'architecture du réseau de neurone optimal, on effectue l'entraînements de plusieurs réseaux de neurones aux configurations variées afin de ne retenir que celui qui minimise la fonction d'erreur pour un temps de calcul raisonnable.\
Les tests ont été faits sur un sous-ensemble de 2 000 trajectoires (40 epochs maximum, arrêt anticipé), en changeant un paramètre à la fois, dans l'ordre d'influence : sens du réseau → learning rate → capacité (nombre de couches × nombre de neurones).

L'erreur plafonne dès 2 couches : le plus gros réseau (3 × 128) ne gagne qu'un peu pour 3,5 fois plus de temps de calcul. On observe aussi que l'erreur sur y reste autour de 4,6 à 5 cm quelle que soit la taille : ce n'est donc pas la capacité qui limite la profondeur.\
On retiendra la configuration suivante :
* Type	LSTM bidirectionnel
* Couches : 2
* Neurones par couche : 64
* Optimiseur	Adam, learning rate 3e-3
* Taille de batch	32
* Gradient clipping	1,0
* Arrêt anticipé	patience de 10 epochs (100 epochs maximum)

## Références
* A. Erades, R. Vuillemot. How Camera Angle Impact Table Tennis Ball Bounce Tracking. Sports Physics, 2025. https://eradesaymeric.github.io/pdf/paper_2025_sports_physics.pdf
* A. Frachet, R. Vuillemot, A. Erades (2026). Analyse avancée de matchs de tennis de table [Projet Appliqué de Recherche]. École Centrale de Lyon, laboratoire LIRIS.
* G. Calmet, A. Erades, R. Vuillemot. Exploring Table Tennis Analytics: Domination, Expected Score and Shot Diversity. Machine Learning and Data Mining for Sports Analytics, Turin, 2023. https://hal.science/hal-04240982/file/_MLSA23__Analyzing_Table_Tennis_Games_using_Detailed_Sports_Data.pdf

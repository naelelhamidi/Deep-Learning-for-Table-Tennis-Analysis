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


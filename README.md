# Application TDA pour l'analyse de spores de *Bacillus subtilis*

Ce dépôt contient le code source de l'application développée dans le cadre de mon stage de Master 2.

L'objectif du projet est d'explorer des données issues de spores de *Bacillus subtilis* soumises à différents traitements UV à l'aide de méthodes d'analyse topologique des données (TDA) et d'approches multiblocks.

L'application a été développée en Python avec Dash afin de permettre une exploration interactive des données et des résultats obtenus avec l'algorithme Mapper.

## Fonctionnalités principales

L'interface permet notamment :

- la sélection de la souche et du traitement UV étudiés ;
- l'utilisation séparée des données Raman et O-PTIR ;
- l'analyse conjointe des deux modalités par des méthodes multiblocks ;
- le choix de différentes fonctions de projection ;
- la construction interactive de graphes Mapper ;
- la modification des paramètres de Mapper et de DBSCAN ;
- la visualisation de la composition des nœuds ;
- l'étude de la pureté des nœuds ;
- la sélection interactive de régions du graphe ;
- la visualisation des spectres moyens et différentiels ;
- l'identification des variables contribuant aux projections ;
- la réalisation de tests statistiques de Mann--Whitney avec correction des comparaisons multiples.

## Méthodes disponibles

### Analyses monoblocs

Les données Raman et O-PTIR peuvent être étudiées séparément à l'aide de plusieurs fonctions de projection :

- PCA (Principal Component Analysis) ;
- PLS-DA (Partial Least Squares Discriminant Analysis).

### Analyses multiblocks

Lorsque les données Raman et O-PTIR sont disponibles pour les mêmes observations, plusieurs méthodes permettent de construire une représentation commune :

- MFA (Multiple Factor Analysis) ;
- MB-PCA (Multi-Block Principal Component Analysis) ;
- MB-PLS (Multi-Block Partial Least Squares) ;
- O2-PLS (Two-Block Orthogonal Partial Least Squares) ;
- MCOA (Multiple Co-Inertia Analysis).

Les représentations obtenues peuvent ensuite être utilisées comme fonctions de projection pour l'algorithme Mapper.

## Algorithme Mapper

La construction du graphe repose sur l'algorithme Mapper et utilise notamment :

- un recouvrement défini par le nombre de cubes (`n_cubes`) ;
- un taux de chevauchement (`perc_overlap`) ;
- DBSCAN pour la clusterisation locale ;
- le paramètre `eps` de DBSCAN ;
- le paramètre `min_samples` de DBSCAN.

L'application permet de modifier ces paramètres directement depuis l'interface et d'observer leur influence sur le graphe obtenu.

## Installation

Une version récente de Python est nécessaire.

Les principales bibliothèques utilisées sont :

```text
numpy
pandas
scikit-learn
scipy
statsmodels
dash
dash-cytoscape
plotly
kmapper
prince

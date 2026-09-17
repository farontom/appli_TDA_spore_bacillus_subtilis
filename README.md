# SpectroMapper — Application TDA pour l’analyse de spores de *Bacillus subtilis*

Ce dépôt contient le code source d’une application interactive consacrée à l’exploration de données spectroscopiques acquises sur des spores de *Bacillus subtilis* soumises à différents traitements UV.

Initialement développée dans le cadre d’un stage de Master 2, l’application a ensuite été enrichie afin d’intégrer de nouvelles méthodes de réduction de dimension, des approches multiblocs et différents outils d’analyse statistique.

L’objectif est d’explorer la structure des données Raman et O-PTIR à l’aide de méthodes chimiométriques et d’analyse topologique des données (*Topological Data Analysis*, TDA), notamment grâce à l’algorithme Mapper.

L’application a été développée en Python avec Dash afin de permettre une exploration interactive des données, des projections et des graphes obtenus.

## Fonctionnalités principales

L’interface permet notamment :

* de sélectionner la souche et le traitement UV étudiés ;
* d’utiliser séparément les données Raman et O-PTIR ;
* d’exploiter conjointement les deux modalités à l’aide de méthodes multiblocs ;
* de choisir différentes fonctions de projection ;
* de sélectionner le prétraitement appliqué aux données ;
* de construire interactivement des graphes Mapper ;
* de modifier les paramètres de Mapper et de DBSCAN ;
* de choisir différentes représentations graphiques du réseau ;
* de visualiser la composition des nœuds ;
* d’étudier la pureté des nœuds ;
* de sélectionner interactivement des régions du graphe ;
* de visualiser les spectres moyens et différentiels ;
* d’identifier les variables contribuant aux projections ;
* de sélectionner des régions spectrales d’intérêt ;
* de réaliser des tests statistiques de Mann–Whitney avec correction pour tests multiples.

## Méthodes disponibles

### Analyses monoblocs

Les données Raman et O-PTIR peuvent être étudiées séparément à l’aide de différentes méthodes de projection :

* PCA (*Principal Component Analysis*) ;
* PLS-DA (*Partial Least Squares Discriminant Analysis*) ;
* t-SNE (*t-distributed Stochastic Neighbor Embedding*) ;
* UMAP (*Uniform Manifold Approximation and Projection*).

La PCA et la PLS-DA fournissent des représentations fondées respectivement sur la variance et sur l’information discriminante associée aux groupes étudiés.

t-SNE et UMAP proposent des représentations non linéaires permettant d’explorer plus particulièrement les structures locales présentes dans les données.

### Analyses multiblocs

Lorsque les données Raman et O-PTIR sont disponibles pour les mêmes observations, plusieurs méthodes peuvent être utilisées afin de construire une représentation commune :

* MFA (*Multiple Factor Analysis*) ;
* MB-PCA (*Multi-Block Principal Component Analysis*) ;
* MB-PLS (*Multi-Block Partial Least Squares*) ;
* O2-PLS (*Two-Block Orthogonal Partial Least Squares*) ;
* MCOA (*Multiple Co-Inertia Analysis*).

Ces approches permettent d’exploiter conjointement les informations provenant des deux modalités spectroscopiques.

Les représentations obtenues peuvent ensuite être utilisées comme fonctions de projection pour la construction des graphes Mapper.

## Algorithme Mapper

L’algorithme Mapper permet de construire une représentation topologique des données sous la forme d’un graphe.

Sa construction repose sur :

* une fonction de projection appliquée aux observations ;
* un recouvrement défini par le nombre d’intervalles ou de cubes (`n_cubes`) ;
* un taux de chevauchement entre les intervalles (`perc_overlap`) ;
* une clusterisation locale réalisée avec DBSCAN ;
* le paramètre de voisinage `eps` de DBSCAN ;
* le nombre minimal d’observations `min_samples` de DBSCAN.

Dans le graphe obtenu :

* chaque nœud correspond à un groupe local d’observations ;
* une même observation peut appartenir à plusieurs nœuds en raison du chevauchement de la couverture ;
* une arête relie deux nœuds lorsqu’ils partagent au moins une observation.

L’application permet de modifier directement les paramètres de Mapper et de DBSCAN afin d’observer leur influence sur la structure du graphe.

## Prétraitement des données

Plusieurs options de prétraitement peuvent être appliquées avant la projection :

* données non transformées ;
* centrage des variables ;
* centrage-réduction des variables.

L’utilisateur peut également sélectionner des bandes spectrales spécifiques afin de concentrer l’analyse sur certaines régions d’intérêt.

## Analyse des résultats

L’application propose plusieurs outils pour faciliter l’interprétation des graphes :

* composition des nœuds selon les traitements ou les souches ;
* calcul de la pureté des nœuds ;
* sélection de nœuds ou de régions du graphe ;
* visualisation des observations associées ;
* calcul de spectres moyens ;
* visualisation de spectres différentiels ;
* étude des contributions des variables ;
* tests statistiques de Mann–Whitney ;
* correction des valeurs de *p* pour tenir compte de la multiplicité des tests.

Ces résultats constituent des outils exploratoires et doivent être interprétés au regard des connaissances biologiques et des conditions expérimentales.

## Installation

Une version récente de Python est recommandée.

Les principales bibliothèques utilisées sont :

```text
numpy
pandas
openpyxl
scikit-learn
scipy
statsmodels
dash
dash-cytoscape
plotly
kmapper
prince
umap-learn
```

Les dépendances peuvent être installées avec la commande suivante :

```bash
pip install numpy pandas openpyxl scikit-learn scipy statsmodels dash dash-cytoscape plotly kmapper prince umap-learn
```

## Lancement de l’application

Après avoir téléchargé le dépôt et installé les dépendances, l’application peut être lancée avec :

```bash
python app.py
```

L’adresse locale affichée dans le terminal peut ensuite être ouverte dans un navigateur web. Par défaut, l’application Dash est généralement accessible à l’adresse suivante :

```text
http://127.0.0.1:8050/
```

## Avertissement

L’application est destinée à l’exploration de données spectroscopiques. Les structures mises en évidence par les méthodes de réduction de dimension, la clusterisation ou l’algorithme Mapper ne constituent pas, à elles seules, une validation biologique.

Les résultats doivent donc être interprétés avec prudence et confrontés aux connaissances biologiques ainsi qu’aux conditions expérimentales.

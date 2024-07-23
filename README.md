# inventory_mgt_drl

# Modélisation volet 2

### Contraintes

- 1 hôpital qui est dépanné ne peut pas dépanner des hôpitaux et inversement
- Chaque hôpitaux a une capacité de stockage max
- Les inventaires ne peuvent pas être négatifs.
- Il est possible d'être en sur stock pendant la journée (le surplus de commandes sont calculés en fin de journée)
- La journée se déroule suivant l'ordre suivant : la réception des commandes de la warehouse, dépannage entre hôpitaux, satisfaction de la demande journalière

### Variables du modèle

state : vecteur représentant l'inventory level de chaque hôpital du problème. un vecteur de taille le nombre d'hôpitaux

action : vecteur représentant les différentes actions prises. Il comprend les commandes des hôpitaux à la warehouse ainsi que les échanges entre-hôpitaux

### Explication des fonctions train et replay



### Vérification que le réseau apprend bien

- Si le réseau (l'agent) doit apprendre à remplir correctement la matrice des demandes ( Dij quantité que demande i à j) notamment à comprendre que le remplissage de la diagonale n'a pas d'impact et également qu'un Dij non nul sur la ligne i implique que les Dji sur la colonne i n'ont plus d'impacts

Voici une synthèse de la documentation du code fourni, qui explique les principales variables et méthodes, en mettant l'accent sur les méthodes `train`, `replay`, ainsi que sur les fonctions `process_demands_random_order`, `is_matrix_correct_form` et `create_random_correct_matrix`.

### Classe `Env`

#### Variables principales :
- **num_hospitals** : Nombre d'hôpitaux.
- **max_inventory** : Niveau maximum d'inventaire pour chaque hôpital.
- **max_demand** : Demande maximale pour chaque hôpital.
- **max_steps** : Nombre maximum d'étapes dans un épisode.
- **demands** : Tableau des demandes courantes des hôpitaux.
- **inventory_levels** : Tableau des niveaux d'inventaire courants des hôpitaux.
- **current_step** : Étape actuelle dans l'épisode.
- **count_in_step_function** : Compteur d'appels à la fonction `step`.
- **count_correct_form** : Compteur des matrices correctement formées.

#### Méthodes principales :
- **`__init__`** : Initialise l'environnement avec les paramètres spécifiés.
- **`reset`** : Réinitialise l'environnement, génère de nouvelles demandes et niveaux d'inventaire.
- **`step`** : Effectue une étape dans l'environnement en fonction de l'action donnée.
- **`render`** : Affiche les niveaux d'inventaire et les demandes.
- **`random_distribution_dict`** : Distribue de manière aléatoire une valeur totale entre les indices donnés.
- **`process_exchanges`** : Traite les échanges entre les hôpitaux.
- **`calculate_demand_overstock`** : Calcule la demande non satisfaite et les quantités de surstock.
- **`process_demands_random_order`** : Corrige la forme de la matrice de demande en traitant les hôpitaux dans un ordre aléatoire.
- **`is_matrix_correct_form`** : Vérifie si la matrice de demande est dans la forme correcte.
- **`create_random_correct_matrix`** : Crée une matrice aléatoire de demandes entre hôpitaux dans la forme correcte.

### Classe `Train_DQL`

#### Variables principales :
- **state_size** : Taille de l'état (nombre d'hôpitaux).
- **sub_action_size_1** : Taille de la sous-action 1 (demande maximale à l'entrepôt).
- **sub_action_size_2** : Taille de la sous-action 2 (demande maximale entre hôpitaux).
- **action_dim** : Dimension de l'action (nombre d'hôpitaux plus le produit carré du nombre d'hôpitaux).
- **hidden_dim** : Dimension cachée du réseau de neurones.
- **memory** : Mémoire de répétition pour stocker les expériences.
- **batch_size** : Taille du lot pour l'apprentissage.
- **gamma** : Facteur de réduction pour les récompenses futures.
- **epsilon** : Paramètre d'exploration pour l'epsilon-greedy.
- **epsilon_min** : Valeur minimale de l'epsilon.
- **epsilon_decay** : Taux de décroissance de l'epsilon.
- **policy_net** : Réseau de neurones pour estimer les valeurs Q.
- **target_net** : Réseau de neurones cible pour l'apprentissage.
- **optimizer** : Optimiseur pour l'apprentissage du réseau de neurones.
- **criterion** : Fonction de perte pour l'apprentissage du réseau de neurones.
- **counter** : Compteur pour les étapes d'apprentissage.

#### Méthodes principales :
- **`remember`** : Stocke une expérience dans la mémoire de répétition.
- **`act`** : Sélectionne une action en utilisant le réseau de neurones.
- **`act_greedy`** : Sélectionne une action de manière greedy avec une exploration epsilon-greedy.
- **`replay`** : Effectue une étape d'apprentissage en échantillonnant un lot de la mémoire.
- **`update_target_network`** : Met à jour le réseau cible avec les poids du réseau de politique.
- **`evaluate_agent`** : Évalue l'agent sur plusieurs épisodes.
- **`train`** : Entraîne l'agent sur l'environnement pour un nombre spécifié d'épisodes.

### Explication des méthodes `train` et `replay`

#### Méthode `train`
La méthode `train` est responsable de l'entraînement de l'agent DQN (Deep Q-Network) sur l'environnement. Elle suit les étapes suivantes :
1. Pour chaque épisode :
   - Réinitialiser l'état de l'environnement.
   - Pour chaque étape de l'épisode :
     - Sélectionner une action en utilisant la politique epsilon-greedy.
     - Effectuer l'action dans l'environnement et observer le nouvel état, la récompense et si l'épisode est terminé.
     - Stocker l'expérience (état, action, récompense, nouvel état, fait) dans la mémoire de répétition.
     - Si l'épisode est terminé, mettre à jour le réseau cible et sortir de la boucle.
     - Sinon, effectuer une étape d'apprentissage en appelant `replay`.
   - Évaluer l'agent à intervalles réguliers et afficher la récompense moyenne.

#### Méthode `replay`
La méthode `replay` est responsable de l'apprentissage de l'agent en utilisant les expériences stockées dans la mémoire de répétition. Elle suit les étapes suivantes :
1. Vérifier si la mémoire contient suffisamment d'expériences pour un lot.
2. Si oui, échantillonner un lot d'expériences aléatoires de la mémoire.
3. Pour chaque dimension de l'action :
   - Calculer les valeurs Q actuelles et les valeurs Q suivantes maximales.
   - Calculer les valeurs Q attendues en utilisant l'équation de Bellman.
4. Calculer la perte entre les valeurs Q actuelles et les valeurs Q attendues.
5. Effectuer une étape de rétropropagation pour mettre à jour les poids du réseau de politique.
6. Réduire l'epsilon pour diminuer progressivement l'exploration.

### Conclusion
Le code décrit la création d'un environnement simulé d'échanges de produits entre hôpitaux et l'entraînement d'un agent DQN pour optimiser ces échanges. Les méthodes et fonctions permettent de gérer les demandes et les inventaires des hôpitaux, d'apprendre des expériences passées, et d'évaluer les performances de l'agent.

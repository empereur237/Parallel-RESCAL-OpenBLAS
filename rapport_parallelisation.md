# Rapport : parallelisation du produit matriciel dans RESCAL-ALS

## 1. Objectif

Le programme RESCAL-ALS manipule beaucoup de matrices. Dans les fonctions de mise a jour de l'algorithme, on retrouve souvent des produits comme :

- `A * B` pour deux matrices denses ;
- `X_k * B` quand `X_k` est une matrice creuse stockee au format CSR ;
- `A * X_k` quand la matrice creuse est a droite.

Ces operations coutent cher, car un produit matriciel classique repete beaucoup de multiplications et d'additions. L'objectif de la parallelisation est donc simple : partager le travail entre plusieurs threads pour reduire le temps total d'execution.

Dans ce code, le nombre de threads est donne au lancement :

```bash
./rescal 4
```

Le programme place cette valeur dans la variable globale `g_num_threads`. Les fonctions `dot`, `dot_csr` et `dot_csr1` lisent ensuite cette variable pour creer le bon nombre de threads.

Le fichier `run_threads.sh` automatise les essais :

```bash
./run_threads.sh
```

Par defaut, le script force le recalcul meme si une ligne existe deja dans le CSV. Cela permet de revoir les executions pour chaque nombre de threads. Pour reprendre une campagne sans recalculer les combinaisons deja presentes, on peut lancer :

```bash
FORCE_RERUN=0 ./run_threads.sh
```

Par defaut, il lance :

```text
2, 4, 6, 8, 10 threads
```

Les resultats sont ajoutes dans `rescal_results.csv`, avec une colonne `threads`.

## 2. Rappel : principe du produit matriciel

Pour multiplier deux matrices :

```text
C = A * B
```

si `A` a `M` lignes et `K` colonnes, et si `B` a `K` lignes et `N` colonnes, alors `C` aura `M` lignes et `N` colonnes.

Chaque case de `C` se calcule ainsi :

```text
C[i][j] = A[i][0] * B[0][j]
        + A[i][1] * B[1][j]
        + ...
        + A[i][K-1] * B[K-1][j]
```

Une case de `C` est donc un produit scalaire entre une ligne de `A` et une colonne de `B`.

Le cout theorique est :

```text
M * N * K operations principales
```

Si les matrices sont grandes, ce cout devient rapidement important.

## 3. Methode de parallelisation utilisee dans notre code

### 3.1. Idee generale

Notre methode decoupe les lignes de la matrice resultat `C`.

Exemple avec 4 threads :

```text
Thread 0 : calcule les lignes 0 a 24 de C
Thread 1 : calcule les lignes 25 a 49 de C
Thread 2 : calcule les lignes 50 a 74 de C
Thread 3 : calcule les lignes 75 a 99 de C
```

Chaque thread recoit donc une zone de lignes. Il lit les matrices d'entree `A` et `B`, puis il ecrit uniquement dans ses propres lignes de `C`.

Cette strategie est importante, car elle evite les conflits d'ecriture :

- plusieurs threads peuvent lire les memes donnees sans probleme ;
- chaque thread ecrit dans une partie differente de `C` ;
- il n'y a donc pas besoin de mutex pendant le calcul.

Moins de synchronisation signifie souvent de meilleures performances.

### 3.2. Repartition des lignes

La fonction `distribute_rows` partage les lignes entre les threads. Si le nombre de lignes n'est pas divisible exactement par le nombre de threads, les premieres tranches recoivent une ligne de plus.

Exemple : 10 lignes et 4 threads.

```text
10 / 4 = 2 lignes par thread, reste 2

Thread 0 : 3 lignes
Thread 1 : 3 lignes
Thread 2 : 2 lignes
Thread 3 : 2 lignes
```

Cela garde une repartition assez equilibree.

### 3.3. Produit dense-dense : `dot(A, B)`

La fonction `dot` calcule le produit de deux matrices denses. Elle suit ces etapes :

1. verifier que les dimensions sont compatibles ;
2. creer la matrice resultat `C` ;
3. choisir le nombre de threads reellement utile ;
4. decouper les lignes de `C` ;
5. lancer les threads avec `pthread_create` ;
6. attendre la fin avec `pthread_join` ;
7. retourner `C`.

Le worker utilise par les threads est `thread_block_dot`.

Le calcul n'est pas seulement parallele : il utilise aussi le cache blocking.

Au lieu de parcourir directement toute la matrice, on travaille par blocs de taille `BLOCK_SIZE`, actuellement `32`. L'idee est de garder des petits morceaux de matrices proches du processeur dans le cache. Ainsi, le processeur reutilise plus souvent les donnees deja chargees, au lieu de retourner trop souvent en memoire principale.

Le coeur du calcul suit l'ordre :

```text
ii -> kk -> jj
```

Cela permet de charger `A[ii][kk]` une fois, puis de le reutiliser pour plusieurs colonnes `jj` :

```c
double a_val = A[ii][kk];
for (int jj = j; jj < j_end; jj++) {
    C[ii][jj] += a_val * B[kk][jj];
}
```

Cette forme aide aussi le compilateur a vectoriser la boucle interne quand le programme est compile avec `-O3 -march=native`.

### 3.4. Produit creux-dense : `dot_csr(A, B)`

Dans RESCAL, les matrices du tenseur `X` contiennent beaucoup de zeros. Les stocker et les parcourir comme des matrices denses gaspillerait du temps et de la memoire.

Le code utilise donc le format CSR, pour Compressed Sparse Row.

CSR stocke seulement :

- les valeurs non nulles ;
- les colonnes de ces valeurs ;
- les positions de debut et de fin de chaque ligne.

Pour `dot_csr(A, B)`, la matrice `A` est creuse et `B` est dense.

Le thread parcourt seulement les valeurs non nulles de chaque ligne de `A` :

```text
pour chaque ligne i de A :
    pour chaque valeur non nulle A[i][col] :
        pour chaque colonne j de B :
            C[i][j] += A[i][col] * B[col][j]
```

Avantage : les zeros ne sont jamais multiplies.

Comme pour `dot`, les lignes de `C` sont partagees entre les threads. Chaque thread ecrit dans ses lignes de sortie, donc il n'y a pas de verrou pendant le calcul.

### 3.5. Produit dense-creux : `dot_csr1(A, B)`

Dans `dot_csr1`, la matrice `A` est dense et `B` est creuse.

Le principe reste le meme : on decoupe les lignes de `C`. Pour chaque ligne dense de `A`, le thread parcourt les lignes CSR de `B` et accumule uniquement les colonnes non nulles.

Cette fonction sert notamment dans `_updateR`, par exemple pour calculer `A^T * X_k`.

## 4. Ou la parallelisation intervient dans RESCAL-ALS

RESCAL-ALS repete plusieurs mises a jour :

- mise a jour de `R` dans `_updateR` ;
- mise a jour de `A` dans `_updateA` ;
- calcul du score de reconstruction dans `_compute_fit` ;
- prediction dans `innerfold`.

Ces etapes appellent plusieurs fois `dot`, `dot_csr` et `dot_csr1`. En parallellisant ces fonctions de base, on accelere donc plusieurs parties de l'algorithme sans devoir reecrire toute la logique de RESCAL.

## 5. Comparaison scientifique : notre methode vs OpenBLAS

### 5.1. Notre methode

Notre implementation est une parallelisation manuelle avec `pthread`.

Ses points forts :

- elle est pedagogique : on voit clairement comment les lignes sont partagees ;
- elle est adaptee au projet : elle gere directement nos structures `Matrix` et `CSRMatrix` ;
- elle evite les verrous pendant le calcul principal ;
- elle exploite les matrices creuses CSR, ce qui est tres utile pour les tenseurs relationnels ;
- elle permet de controler facilement le nombre de threads via `./rescal N`.

Ses limites :

- elle ne contient pas tous les micro-optimisations d'une bibliotheque professionnelle ;
- elle ne choisit pas automatiquement le meilleur algorithme selon la taille des matrices ;
- elle peut etre moins performante sur les tres gros produits denses ;
- elle depend de notre organisation memoire `double**`, moins favorable aux optimisations bas niveau qu'un grand tableau contigu.

### 5.2. OpenBLAS

OpenBLAS est une bibliotheque optimisee pour les operations d'algebre lineaire. Pour le produit matriciel dense, elle utilise generalement des routines de type GEMM.

Ses points forts :

- elle est tres optimisee pour les processeurs modernes ;
- elle utilise des kernels vectorises adaptes a l'architecture CPU ;
- elle gere tres bien les caches, les registres et les instructions SIMD ;
- elle peut paralleliser automatiquement les gros produits ;
- elle est souvent beaucoup plus rapide pour les produits dense-dense de grande taille.

Ses limites dans notre contexte :

- elle demande souvent des matrices stockees dans un format contigu compatible BLAS ;
- notre structure `Matrix` utilise `double**`, donc il faudrait convertir ou reorganiser les donnees ;
- BLAS standard cible surtout les matrices denses ;
- pour les matrices creuses CSR, il faut une autre interface ou une autre bibliotheque sparse ;
- si OpenBLAS et notre code `pthread` lancent tous les deux des threads, on peut creer trop de threads et ralentir le programme.

### 5.3. Difference centrale

La difference principale est le niveau d'optimisation.

Notre methode travaille au niveau algorithmique simple :

```text
Je decoupe les lignes, je donne chaque groupe de lignes a un thread.
```

OpenBLAS travaille au niveau bas niveau :

```text
Je choisis des micro-kernels optimises pour le processeur, j'organise les donnees
pour le cache et les registres, puis je vectorise fortement le calcul.
```

Notre methode est donc plus facile a expliquer et a adapter au code RESCAL. OpenBLAS est souvent plus rapide pour les produits denses purs, mais il impose plus de contraintes sur le format memoire et l'integration.

## 6. Tableau comparatif

| Critere | Notre methode pthread | OpenBLAS |
|---|---|---|
| Objectif principal | Controler et expliquer la parallelisation | Performance maximale sur l'algebre lineaire |
| Produit dense-dense | Oui, avec decoupage par lignes et cache blocking | Oui, tres optimise via GEMM |
| Matrices creuses CSR | Oui, directement dans `dot_csr` et `dot_csr1` | Pas via BLAS dense standard |
| Controle des threads | Direct avec `./rescal N` | Via variables comme `OPENBLAS_NUM_THREADS` |
| Niveau pedagogique | Tres lisible | Plus difficile a expliquer car tres optimise |
| Performance dense pure | Correcte | Generalement excellente |
| Integration avec ce projet | Native | Necessite adaptation du stockage memoire |
| Risque d'oversubscription | Faible si seul notre pthread est utilise | Possible si OpenBLAS et pthread parallelisent en meme temps |

## 7. Comment presenter cela simplement

Pour expliquer devant des debutants, on peut utiliser une image simple :

1. Une matrice resultat `C` est comme un cahier avec plusieurs lignes a remplir.
2. Au lieu de donner tout le cahier a une seule personne, on partage les pages.
3. Chaque thread est une personne qui remplit ses propres lignes.
4. Comme personne n'ecrit sur les memes lignes, il n'y a pas de dispute.
5. A la fin, on rassemble le cahier complet : c'est la matrice `C`.

Pour les matrices creuses, on ajoute :

1. Beaucoup de cases valent zero.
2. Multiplier par zero ne sert a rien.
3. Le format CSR garde seulement les cases utiles.
4. Les threads travaillent donc sur les vraies informations, pas sur les zeros.

## 8. Lecture des resultats

Le fichier `rescal_results.csv` contient une ligne par combinaison :

```text
dataset;rank;maxIter;conv;lambda_A;lambda_R;nnz;import_time_s;auc_pr_mean;auc_pr_std;total_time_s;threads
```

La colonne la plus importante pour comparer la parallelisation est `total_time_s`.

Pour chaque dataset et chaque configuration, on compare :

```text
temps avec 2 threads
temps avec 4 threads
temps avec 6 threads
temps avec 8 threads
temps avec 10 threads
```

On peut ensuite calculer l'acceleration :

```text
speedup = temps_reference / temps_parallelise
```

Exemple :

```text
temps avec 2 threads  = 100 s
temps avec 8 threads  = 35 s
speedup = 100 / 35 = 2.86
```

Cela signifie que l'execution avec 8 threads est environ 2.86 fois plus rapide que l'execution de reference a 2 threads.

## 9. Conclusion

La parallelisation actuelle est basee sur une idee robuste : partager les lignes de la matrice resultat entre les threads. Cette methode evite les conflits d'ecriture, limite la synchronisation et reste facile a expliquer.

Pour notre projet RESCAL-ALS, elle a aussi un avantage important : elle gere a la fois les produits denses et les produits avec matrices creuses CSR. OpenBLAS serait probablement plus rapide pour les grands produits denses, mais il faudrait adapter le format memoire et faire attention a la gestion des threads.

Ainsi, notre implementation est un bon compromis pour ce travail : elle est claire, controlable, compatible avec les structures du projet, et suffisamment scientifique pour mesurer l'effet du nombre de threads dans `rescal_results.csv`.

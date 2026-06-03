# Rapport sur le produit matriciel OpenBLAS dans le projet

## 1. Idee generale

Dans ce projet RESCAL-ALS, le produit matriciel dense n'est plus calcule manuellement avec trois boucles `for`. Il est delegue a OpenBLAS via la fonction `cblas_dgemm`, appelee dans la fonction `dot()` du fichier `utiles.c`.

La fonction `dot(A, B)` est donc la fonction centrale du projet pour calculer un produit matriciel dense :

```text
C = A x B
C[i][j] = somme_k A[i][k] * B[k][j]
```

Avant de lancer le calcul, le code verifie que les dimensions sont compatibles :

```text
A->cols == B->rows
```

Puis il cree une matrice resultat `C` de taille :

```text
A->rows x B->cols
```

## 2. Appel OpenBLAS utilise

Dans `utiles.c`, la fonction `dot()` appelle OpenBLAS avec :

```c
cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    n,
    p,
    m,
    1.0,
    A->data[0],
    A->cols,
    B->data[0],
    B->cols,
    0.0,
    C->data[0],
    C->cols
);
```

Cet appel signifie :

- `CblasRowMajor` : les matrices sont stockees ligne par ligne en memoire.
- `CblasNoTrans` pour `A` : OpenBLAS ne transpose pas `A`.
- `CblasNoTrans` pour `B` : OpenBLAS ne transpose pas `B`.
- `n` : nombre de lignes de `A`.
- `p` : nombre de colonnes de `B`.
- `m` : dimension commune entre `A` et `B`.
- `alpha = 1.0` : coefficient applique a `A x B`.
- `beta = 0.0` : l'ancien contenu de `C` est ignore.

La formule BLAS generale est :

```text
C = alpha * A * B + beta * C
```

Dans ce projet, comme `alpha = 1.0` et `beta = 0.0`, cela revient exactement a :

```text
C = A * B
```

## 3. Organisation memoire des matrices

Les matrices denses sont creees avec `init_Matrix()` dans `utiles.c`.

Le point important est que les donnees sont allouees dans un bloc contigu :

```c
matrix->data[0] = calloc(rows * cols, sizeof(double));
matrix->data[i] = matrix->data[0] + i * cols;
```

Cela veut dire que meme si le programme utilise l'ecriture pratique :

```c
matrix->data[i][j]
```

les valeurs sont physiquement rangees de facon continue en memoire.

C'est tres important pour OpenBLAS, car une memoire contigue permet :

- une meilleure utilisation du cache processeur ;
- une lecture plus rapide des lignes ;
- une meilleure vectorisation ;
- moins de surcout d'acces memoire.

On peut donc expliquer que le projet prepare correctement les matrices pour que `cblas_dgemm` puisse travailler efficacement.

## 4. Ou OpenBLAS intervient dans RESCAL-ALS

OpenBLAS intervient chaque fois que le programme appelle `dot()` avec deux matrices denses.

Dans la mise a jour des matrices relationnelles `R`, le code calcule notamment :

```text
A^T A
A^T X_k A
Z_inv * (A^T X_k A)
R_k = Z_inv * (A^T X_k A) * Z_inv
```

Les parties denses comme :

```c
AtA = dot(At, A);
AtXkA = dot(AtXk, A);
left = dot(Z_inv, AtXkA);
Rk = dot(left, Z_inv);
```

passent par OpenBLAS.

Dans la mise a jour de `A`, on retrouve aussi plusieurs produits denses :

```text
A * R_k^T
X_k^T A * R_k
A^T A * R_k^T
R_k * A^T A * R_k^T
```

Les produits dense x dense appellent `dot()`, donc `cblas_dgemm`.

Dans la prediction, le modele reconstruit chaque tranche avec :

```text
P_k = A * R_k * A^T
```

Le code fait :

```c
temp2 = dot(&R->slices[k], temp1);
temp3 = dot(A, temp2);
```

Ces deux produits sont aussi calcules avec OpenBLAS.

## 5. Attention : tout le produit matriciel n'utilise pas OpenBLAS

Il faut bien distinguer deux cas.

### Cas 1 : dense x dense

Quand les deux matrices sont denses, le projet utilise :

```c
dot(A, B)
```

Cette fonction appelle OpenBLAS avec `cblas_dgemm`.

### Cas 2 : produits avec matrices creuses CSR

Quand une matrice est creuse, le projet utilise des fonctions specifiques :

```c
dot_csr(CSRMatrix* A, Matrix* B)
dot_csr1(Matrix* A, CSRMatrix* B)
```

Ces fonctions ne passent pas par OpenBLAS. Elles sont codees avec des boucles, car elles exploitent le format CSR.

Le format CSR stocke uniquement :

- les valeurs non nulles ;
- les indices de colonnes ;
- les positions de debut de ligne.

C'est tres utile pour les graphes de connaissances, car les tenseurs comme FB15k-237 sont tres creux. Il serait couteux et inutile de parcourir tous les zeros.

On peut donc defendre le choix ainsi :

```text
Le projet utilise OpenBLAS pour les calculs denses, et garde un calcul CSR specifique pour les donnees creuses afin d'eviter de multiplier inutilement des zeros.
```

## 6. Parametres qui influencent le produit matriciel

### 6.1 Nombre de threads

Dans `dot()`, le code recupere le nombre maximal de threads OpenMP :

```c
int max_threads = omp_get_max_threads();
openblas_set_num_threads(max_threads);
```

Cela fixe le nombre de threads utilises par OpenBLAS.

En pratique, ce nombre peut etre influence par l'environnement, par exemple avec :

```bash
OMP_NUM_THREADS=4 ./rescal
```

Plus il y a de threads, plus OpenBLAS peut paralleliser le calcul. Mais au-dela d'un certain point, ajouter des threads peut devenir moins efficace a cause des couts de synchronisation et de la pression memoire.

### 6.2 Dimensions des matrices

Pour un produit :

```text
A : n x m
B : m x p
C : n x p
```

le cout theorique est :

```text
O(n * m * p)
```

Plus `n`, `m` et `p` sont grands, plus le calcul est lourd. C'est justement dans ce cas qu'OpenBLAS devient interessant.

### 6.3 Rang RESCAL

Le parametre `rank` influence directement la taille des matrices du modele.

Si :

```text
n = nombre d'entites
r = rank
```

alors :

```text
A   : n x r
R_k : r x r
A^T A : r x r
```

Quand `rank` augmente, les produits deviennent plus couteux, car les matrices internes sont plus grandes.

Dans la configuration actuelle du projet, on trouve par exemple :

```c
{ 200, 20, 1e-6, 0.1, 0.1, 5.0 }
```

Ici :

- `rank = 200` ;
- `maxIter = 20` ;
- `conv = 1e-6` ;
- `lambda_A = 0.1` ;
- `lambda_R = 0.1` ;
- `lambda_Z = 5.0`.

### 6.4 Nombre de relations ou tranches du tenseur

Dans RESCAL, il existe une matrice `R_k` pour chaque relation.

Donc plus il y a de relations, plus il y a de tranches, et plus le programme repete les produits matriciels.

Cela influence fortement le temps total d'execution.

### 6.5 Nombre d'elements non nuls

Le parametre `nnz` represente le nombre d'elements non nuls du tenseur.

Il influence surtout les fonctions CSR :

```c
dot_csr()
dot_csr1()
```

Si le tenseur est tres creux, ces fonctions sont avantageuses parce qu'elles ne parcourent que les valeurs utiles.

### 6.6 Nombre d'iterations

Le parametre `maxIter` controle le nombre maximal d'iterations ALS.

A chaque iteration, le programme met a jour :

- la matrice `A` ;
- les matrices `R_k` ;
- eventuellement les matrices `Z`.

Chaque iteration contient plusieurs produits matriciels. Donc plus `maxIter` est grand, plus le nombre total d'appels a OpenBLAS augmente.

### 6.7 Critere de convergence

Le parametre `conv` permet d'arreter l'algorithme lorsque la variation du fit devient faible :

```c
if (itr > 0 && fitchange < conv) break;
```

Un seuil plus strict peut provoquer plus d'iterations. Un seuil plus large peut arreter plus tot.

### 6.8 Options de compilation

Le `makefile` compile avec :

```makefile
-O3 -march=native -fopenmp
-lopenblas -lm -lpthread
```

Les elements importants sont :

- `-lopenblas` : lie le programme a OpenBLAS ;
- `-O3` : active des optimisations fortes du compilateur ;
- `-march=native` : adapte le code au processeur de la machine ;
- `-fopenmp` : active OpenMP, dont le nombre de threads est utilise pour configurer OpenBLAS.

## 7. Comparaison avec l'ancienne version

Le projet contient aussi une ancienne version dans `V1_utiles.c`.

Dans cette version, le produit matriciel dense etait code manuellement avec :

- des threads `pthread` ;
- un decoupage des lignes ;
- un blocage cache avec `BLOCK_SIZE` ;
- une fonction interne `thread_block_dot`.

Cette ancienne approche est interessante pedagogiquement, car elle montre comment on peut paralleliser manuellement un produit matriciel.

Mais la version actuelle avec OpenBLAS est plus robuste pour la performance, car OpenBLAS utilise des noyaux optimises pour le processeur, le cache et la vectorisation.

## 8. Phrase simple pour la soutenance

Tu peux dire :

```text
Dans mon projet, les produits matriciels denses sont centralises dans la fonction dot().
Cette fonction ne fait pas le produit manuellement : elle appelle cblas_dgemm d'OpenBLAS.
OpenBLAS calcule C = A x B avec des noyaux optimises, en exploitant le cache,
la vectorisation et plusieurs threads. Les matrices sont stockees dans un bloc
memoire contigu, ce qui permet a OpenBLAS de travailler efficacement. En revanche,
pour les matrices creuses du tenseur, j'utilise des fonctions CSR specifiques afin
d'eviter de parcourir les zeros. Donc le projet combine OpenBLAS pour le dense et
CSR manuel pour le creux.
```

## 9. Questions-reponses possibles

### Question : Pourquoi utiliser OpenBLAS ?

Reponse :

```text
Parce que le produit matriciel dense est une operation tres couteuse.
OpenBLAS fournit dgemm, une routine BLAS niveau 3 tres optimisee, plus performante
qu'une triple boucle classique. Elle exploite le cache, les instructions vectorielles
et le parallelisme.
```

### Question : Est-ce que tous les produits matriciels utilisent OpenBLAS ?

Reponse :

```text
Non. Les produits dense x dense utilisent OpenBLAS via dot().
Les produits impliquant des matrices creuses CSR utilisent dot_csr() et dot_csr1(),
car il est plus efficace de ne parcourir que les elements non nuls.
```

### Question : Quel est le role de cblas_dgemm ?

Reponse :

```text
cblas_dgemm calcule le produit matriciel general en double precision.
Dans ce projet, il calcule C = A x B, car alpha vaut 1.0 et beta vaut 0.0.
```

### Question : Pourquoi les matrices sont en RowMajor ?

Reponse :

```text
Parce que le programme C stocke les matrices ligne par ligne.
Le parametre CblasRowMajor indique a OpenBLAS comment lire correctement les donnees.
```

### Question : Quels parametres influencent les performances ?

Reponse :

```text
Les performances dependent du nombre de threads, de la taille des matrices,
du rang RESCAL, du nombre de relations, du nombre d'elements non nuls,
du nombre d'iterations, du critere de convergence et des options de compilation.
```

### Question : Pourquoi ne pas convertir toutes les matrices CSR en dense pour utiliser OpenBLAS ?

Reponse :

```text
Parce que les donnees de graphe de connaissances sont tres creuses.
Les convertir en dense consommerait beaucoup plus de memoire et ferait beaucoup
de calculs inutiles sur des zeros. Le CSR permet de ne traiter que les valeurs utiles.
```

## 10. Conclusion

Le produit matriciel OpenBLAS dans ce projet est realise par la fonction `dot()`, qui appelle `cblas_dgemm`.

Cette implementation permet d'accelerer les produits dense x dense dans RESCAL-ALS, notamment lors des mises a jour de `A`, de `R` et lors de la prediction.

Le projet combine donc deux strategies :

- OpenBLAS pour les calculs denses rapides ;
- CSR manuel pour les calculs impliquant des matrices creuses.

Ce choix est coherent avec la nature du probleme : RESCAL manipule des matrices denses internes, mais les donnees d'origine sont tres creuses.

/*
 * Auteur      : Projet RESCAL-ALS
 * Date        : 2026
 * Description : mises à jour ALS et interface principale RESCAL.
 */

#ifndef RESCAL_H
#define RESCAL_H

#include "utiles.h"
#include "svd.h"

double _compute_fit(CSR3DTensor* X,Tensor3D* X1, Matrix* A, Tensor3D* R, double lambda_A, double lambda_R);

Tensor3D* _updateR(CSR3DTensor* X, Matrix* A, double lmbdaR, int rank);

Tensor3D* _updateZ(CSR3DTensor* P, Matrix* A, double lambda_V);

Matrix* _updateA(CSR3DTensor* X, Matrix* A, Tensor3D* R, CSR3DTensor* P, Tensor3D* Z, double lambda_A, int rank);

resultat rescal_als(Tensor3D* X1, int rank, char* init, int maxIter, double conv, double lambda_A, double lambda_R, double lambda_Z, Tensor3D* P1, int dtype);


#endif


#include "svd.h"

SVDResult* MatDecompSVD(Matrix* A) {
    int m = A->rows;
    int n = A->cols;
    Matrix* a = copy_Matrix(A);
    int mn = m < n ? m : n;
    double* s = (double*)malloc(mn * sizeof(double));
    Matrix* u = init_Matrix(m, n);
    Matrix* v = init_Matrix(n, n);
    double* z = (double*)malloc(n * sizeof(double));
    double* scratch = (double*)malloc(m * sizeof(double));

    int nct = (m - 1) < n ? (m - 1) : n;
    int nrt = ((n - 2) < m ? (n - 2) : m) > 0 ? ((n - 2) < m ? (n - 2) : m) : 0;

    for (int k = 0; k < (nct > nrt ? nct : nrt); ++k) {
        if (k < nct) {
            s[k] = 0;
            for (int i = k; i < m; ++i)
                s[k] = LocalHypot(s[k], a->data[i][k]);

            if (s[k] != 0.0) {
                if (a->data[k][k] < 0.0)
                    s[k] = -s[k];

                for (int i = k; i < m; ++i)
                    a->data[i][k] /= s[k];

                a->data[k][k] += 1.0;
            }

            s[k] = -s[k];
        }

        for (int j = k + 1; j < n; ++j) {
            if ((k < nct) && (s[k] != 0.0)) {
                double t = 0;
                for (int i = k; i < m; ++i)
                    t += a->data[i][k] * a->data[i][j];
                t = -t / a->data[k][k];
                for (int i = k; i < m; ++i)
                    a->data[i][j] += t * a->data[i][k];
            }
            z[j] = a->data[k][j];
        }

        if (k < nct) {
            for (int i = k; i < m; ++i)
                u->data[i][k] = a->data[i][k];
        }

        if (k < nrt) {
            z[k] = 0;
            for (int i = k + 1; i < n; ++i) {
                z[k] = LocalHypot(z[k], z[i]);
            }

            if (z[k] != 0.0) {
                if (z[k + 1] < 0.0)
                    z[k] = -z[k];

                for (int i = k + 1; i < n; ++i)
                    z[i] /= z[k];

                z[k + 1] += 1.0;
            }

            z[k] = -z[k];
            if ((k + 1 < m) && (z[k] != 0.0)) {
                for (int i = k + 1; i < m; ++i)
                    scratch[i] = 0.0;

                for (int j = k + 1; j < n; ++j)
                    for (int i = k + 1; i < m; ++i)
                        scratch[i] += z[j] * a->data[i][j];

                for (int j = k + 1; j < n; ++j) {
                    double t = -z[j] / z[k + 1];
                    for (int i = k + 1; i < m; ++i)
                        a->data[i][j] += t * scratch[i];
                }
            }

            for (int i = k + 1; i < n; ++i)
                v->data[i][k] = z[i];
        }
    }

    int p = n < (m + 1) ? n : (m + 1);
    if (nct < n) s[nct] = a->data[nct][nct];
    if (m < p) s[p - 1] = 0.0;
    if (nrt + 1 < p) z[nrt] = a->data[nrt][p - 1];
    z[p - 1] = 0.0;

    for (int j = nct; j < mn; ++j) {
        for (int i = 0; i < m; ++i)
            u->data[i][j] = 0.0;
        u->data[j][j] = 1.0;
    }

    for (int k = nct - 1; k >= 0; --k) {
        if (s[k] != 0.0) {
            for (int j = k + 1; j < mn; ++j) {
                double t = 0;
                for (int i = k; i < m; ++i)
                    t += u->data[i][k] * u->data[i][j];

                t = -t / u->data[k][k];
                for (int i = k; i < m; ++i)
                    u->data[i][j] += t * u->data[i][k];
            }

            for (int i = k; i < m; ++i)
                u->data[i][k] = -u->data[i][k];

            u->data[k][k] = 1 + u->data[k][k];
            for (int i = 0; i < k - 1; ++i)
                u->data[i][k] = 0.0;
        } else {
            for (int i = 0; i < m; ++i)
                u->data[i][k] = 0.0;
            u->data[k][k] = 1.0;
        }
    }

    for (int k = n - 1; k >= 0; --k) {
        if ((k < nrt) && (z[k] != 0.0)) {
            for (int j = k + 1; j < n; ++j) {
                double t = 0;
                for (int i = k + 1; i < n; ++i)
                    t += v->data[i][k] * v->data[i][j];

                t = -t / v->data[k + 1][k];
                for (int i = k + 1; i < n; ++i)
                    v->data[i][j] += t * v->data[i][k];
            }
        }

        for (int i = 0; i < n; ++i)
            v->data[i][k] = 0.0;
        v->data[k][k] = 1.0;
    }

    int pp = p - 1;
    int iter = 0;
    double eps = 1.0e-32;
    while (p > 0) {
        if (iter >= 100)
            printf("Warn: no convergence\n");
        int k;
        int branch;

        for (k = p - 2; k >= -1; --k) {
            if (k == -1)
                break;

            if (fabs(z[k]) <= eps * (fabs(s[k]) + fabs(s[k + 1]))) {
                z[k] = 0.0;
                break;
            }
        }

        if (k == p - 2)
            branch = 4;
        else {
            int ks;
            for (ks = p - 1; ks >= k; --ks) {
                if (ks == k)
                    break;

                double t;
                double tmp;
                if (ks != p)
                    tmp = fabs(z[ks]);
                else
                    tmp = 0.0;
                if (ks != k + 1)
                    t = tmp + fabs(z[ks - 1]);
                else
                    t = tmp + 0.0;

                if (fabs(s[ks]) <= eps * t) {
                    s[ks] = 0.0;
                    break;
                }
            }

            if (ks == k)
                branch = 3;
            else if (ks == p - 1)
                branch = 1;
            else {
                branch = 2;
                k = ks;
            }
        }

        ++k;

        if (branch == 1) {
            double f = z[p - 2];
            z[p - 2] = 0.0;
            for (int j = p - 2; j >= k; --j) {
                double t = LocalHypot(s[j], f);
                double cs = s[j] / t;
                double sn = f / t;
                s[j] = t;
                if (j != k) {
                    f = -sn * z[j - 1];
                    z[j - 1] = cs * z[j - 1];
                }

                for (int i = 0; i < n; ++i) {
                    t = cs * v->data[i][j] + sn * v->data[i][p - 1];
                    v->data[i][p - 1] = -sn * v->data[i][j] + cs * v->data[i][p - 1];
                    v->data[i][j] = t;
                }
            }
        } else if (branch == 2) {
            double f = z[k - 1];
            z[k - 1] = 0.0;
            for (int j = k; j < p; ++j) {
                double t = LocalHypot(s[j], f);
                double cs = s[j] / t;
                double sn = f / t;
                s[j] = t;
                f = -sn * z[j];
                z[j] = cs * z[j];

                for (int i = 0; i < m; ++i) {
                    t = cs * u->data[i][j] + sn * u->data[i][k - 1];
                    u->data[i][k - 1] = -sn * u->data[i][j] + cs * u->data[i][k - 1];
                    u->data[i][j] = t;
                }
            }
        } else if (branch == 3) {
            double max1 = fmax(fabs(s[p - 1]), fabs(s[p - 2]));
            double max2 = fmax(max1, fabs(z[p - 2]));
            double max3 = fmax(max2, fabs(s[k]));
            double max4 = fmax(max3, fabs(z[k]));
            double scale = max4;

            double sp = s[p - 1] / scale;
            double spm1 = s[p - 2] / scale;
            double epm1 = z[p - 2] / scale;
            double sk = s[k] / scale;
            double zk = z[k] / scale;
            double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
            double c = (sp * epm1) * (sp * epm1);
            double shift = 0.0;
            if ((b != 0.0) || (c != 0.0)) {
                shift = sqrt(b * b + c);
                if (b < 0.0)
                    shift = -shift;
                shift = c / (b + shift);
            }

            double f = (sk + sp) * (sk - sp) + shift;
            double g = sk * zk;

            for (int j = k; j < p - 1; ++j) {
                double t = LocalHypot(f, g);
                double cs = f / t;
                double sn = g / t;
                if (j != k)
                    z[j - 1] = t;
                f = cs * s[j] + sn * z[j];
                z[j] = cs * z[j] - sn * s[j];
                g = sn * s[j + 1];
                s[j + 1] = cs * s[j + 1];

                for (int i = 0; i < n; ++i) {
                    t = cs * v->data[i][j] + sn * v->data[i][j + 1];
                    v->data[i][j + 1] = -sn * v->data[i][j] + cs * v->data[i][j + 1];
                    v->data[i][j] = t;
                }

                t = LocalHypot(f, g);
                cs = f / t;
                sn = g / t;
                s[j] = t;
                f = cs * z[j] + sn * s[j + 1];
                s[j + 1] = -sn * z[j] + cs * s[j + 1];
                g = sn * z[j + 1];
                z[j + 1] = cs * z[j + 1];

                if (j < m - 1) {
                    for (int i = 0; i < m; ++i) {
                        double t = cs * u->data[i][j] + sn * u->data[i][j + 1];
                        u->data[i][j + 1] = -sn * u->data[i][j] + cs * u->data[i][j + 1];
                        u->data[i][j] = t;
                    }
                }
            }

            z[p - 2] = f;
            ++iter;
        } else if (branch == 4) {
            if (s[k] <= 0.0) {
                if (s[k] < 0.0)
                    s[k] = -s[k];
                else
                    s[k] = 0.0;

                for (int i = 0; i <= pp; ++i)
                    v->data[i][k] = -v->data[i][k];
            }

            while (k < pp) {
                if (s[k] >= s[k + 1])
                    break;

                double t = s[k];
                s[k] = s[k + 1];
                s[k + 1] = t;

                if (k < n - 1) {
                    for (int i = 0; i < n; ++i) {
                        t = v->data[i][k + 1];
                        v->data[i][k + 1] = v->data[i][k];
                        v->data[i][k] = t;
                    }
                }

                if (k < m - 1) {
                    for (int i = 0; i < m; ++i) {
                        t = u->data[i][k + 1];
                        u->data[i][k + 1] = u->data[i][k];
                        u->data[i][k] = t;
                    }
                }

                ++k;
            }

            iter = 0;
            --p;
        }
    }

    SVDResult* result = (SVDResult*)malloc(sizeof(SVDResult));
    result->U = *u;
    result->S = s;
    result->Vt = *v;

    free(u);
    free(v);
    free(z);
    free(scratch);
    free_Matrix(a);

    return result;
}

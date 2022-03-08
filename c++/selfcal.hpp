//#include "mfista.h"

#include <time.h>

#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdlib>
#include <cmath>

#define MAXITER   50000
#define MINITER   100
#define EPS       1.0e-5
#define RHOSTEP   100

using namespace std;
using namespace Eigen;

#ifdef __cplusplus
extern "C" {
#endif

// mfista_nufft_lib

double self_calibration(double *vis_r, double *vis_i, double *vis_std,
                        double *y_r, double *y_i,
                        double *ginit_r, double *ginit_i,
                        int **gid2vid_tbl, int **vid2gid_st, double *time_tbl,
                        int M, int Gnum, int Stnum, int Tnum,
                        double lambda_1, double lambda_2, double rho_init,
                        int maxiter, double eps,
                        double *gain_r, double *gain_i);

void modify_visibility(double *vis_r, double *vis_i,
                       double *gain_r, double *gain_i,
                       int M, int Gnum, int **vid2gid_st,
                       double *newvis_r, double *newvis_i);

#ifdef __cplusplus
}
#endif

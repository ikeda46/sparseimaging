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

#include <fftw3.h>

#define MAXITER   50000
#define MINITER   100
#define FGPITER   100
#define TD        50
#define ETA       1.1
#define EPS       1.0e-5

#define NU_SIGN -1
// for high precision
// #define MSP 12

// for low precision
#define MSP 6

// 2022/01/25
#define MSP2 12	// MSP*2
#define MSP4 24	// MSP*4

using namespace std;
using namespace Eigen;

#ifdef __cplusplus
extern "C" {
#endif

struct RESULT{
  int M;
  int N;
  int NX;
  int NY;
  int N_active;
  int maxiter;
  int ITER;
  int nonneg;
  double lambda_l1;
  double lambda_tv;
  double lambda_tsv;
  double sq_error;
  double mean_sq_error;
  double l1cost;
  double tvcost;
  double tsvcost;
  double finalcost;
  double comp_time;
  double *residual;
  double Lip_const;
};

#ifdef __cplusplus
}
#endif

struct IO_FNAMES{
  unsigned int fft;
  char *fft_fname;
  char *v_fname;
  char *A_fname;
  char *in_fname;
  char *out_fname;
};

// mfista_io

void init_result(struct IO_FNAMES *mfista_io,
		 struct RESULT *mfista_result);

void cout_result(char *fname,
		 struct IO_FNAMES *mfista_io,
		 struct RESULT *mfista_result);

void write_result(ostream *ofs, char *fname, struct IO_FNAMES *mfista_io,
		  struct RESULT *mfista_result);

// soft thresholding

double calc_Q_part(VectorXd &xvec1, VectorXd &xvec2,
		   double c, VectorXd &AyAz, VectorXd &buf_vec);


void soft_threshold_box(VectorXd &nvec, VectorXd &vec, double eta,
			int box_flag, VectorXd &box);

void soft_threshold_nonneg_box(VectorXd &nvec, VectorXd &vec, double eta,
			       int box_flag, VectorXd &box);

double TSV(int Nx, int Ny, VectorXd &xvec, VectorXd &buf_diff);

void d_TSV(VectorXd &dvec, int Nx, int Ny, VectorXd &xvec);

void get_current_time(struct timespec *t);


#ifdef __cplusplus
extern "C" {
#endif
// mfista_nufft_lib

void mfista_imaging_core_nufft(double *u_dx, double *v_dy,
			       double *vis_r, double *vis_i, double *vis_std,
			       int M, int Nx, int Ny, int maxiter, double eps,
			       double lambda_l1, double lambda_tv, double lambda_tsv,
			       double cinit, double *xinit, double *xout,
			       int nonneg_flag, int box_flag, float *cl_box,
			       struct RESULT *mfista_result);

// mfista_fft_lib

void mfista_imaging_core_fft(int *u_idx, int *v_idx,
			     double *y_r, double *y_i, double *noise_stdev,
			     int M, int Nx, int Ny, int maxiter, double eps,
			     double lambda_l1, double lambda_tv, double lambda_tsv,
			     double cinit, double *xinit, double *xout,
			     int nonneg_flag, unsigned int fftw_plan_flag,
			     int box_flag, float *cl_box,
			     struct RESULT *mfista_result);

#ifdef __cplusplus
}
#endif

#include "mfista.hpp"

#include <time.h>

#ifdef __APPLE__
#include <sys/time.h>
#endif

// mfista_tools

double calc_Q_part(VectorXd &xvec1, VectorXd &xvec2,
		   double c, VectorXd &AyAz, VectorXd &buf_vec)
{
  #ifdef _OPENMP
  double term = 0;
  #pragma omp parallel for reduction(+:term)
  for (int i = 0; i < xvec1.size(); ++i)
  {
    double tmp = xvec1[i] - xvec2[i];
    buf_vec[i] = tmp;
    double term1 = tmp * AyAz[i];
    double term2 = tmp * tmp;
    term += - term1 + c * term2 / 2;
  }
  return term;
  #else
  double term1, term2;

  // x1 - x2
  buf_vec = xvec1 - xvec2;
  // (x1 - x2)'A'(y - A x2)
  term1 = buf_vec.dot(AyAz);
  // (x1 - x2)'(x1 - x2)
  term2 = buf_vec.squaredNorm();

  return(-term1+c*term2/2);
  #endif
}

// soft thresholding

void soft_threshold(VectorXd &nvec, VectorXd &vec, double eta)
{
  int i;

  for(i = 0; i < vec.size(); i++){
    if(vec(i) >= eta)       nvec(i) = vec(i) - eta;
    else if(vec(i) <=- eta) nvec(i) = vec(i) + eta;
    else                    nvec(i) = 0;
  }
}

void soft_threshold_box(VectorXd &nvec, VectorXd &vec, double eta,
			int box_flag, VectorXd &box)
{
  soft_threshold(nvec, vec, eta);
  if(box_flag == 1) nvec = nvec.array()*box.array();
}

void soft_threshold_nonneg(VectorXd &nvec, VectorXd &vec, double eta)
{
  //  nvec = vec.array()-eta;
  //  nvec = nvec.unaryExpr([](double x) { return (x > 0) ? x : 0.0; });
  int i;

  for(i = 0; i < vec.size(); i++){
    if(vec(i) >= eta)       nvec(i) = vec(i) - eta;
    else                    nvec(i) = 0;
  }
}

void soft_threshold_nonneg_box(VectorXd &nvec, VectorXd &vec, double eta,
			       int box_flag, VectorXd &box)
{
  int i;

  if(box_flag == 1){
    for(i = 0; i < vec.size(); i++){
      if(box(i) == 0) nvec(i) = 0;
      else if (vec(i) >= eta) nvec(i) = vec(i) - eta;
      else                    nvec(i) = 0;
    }
  }
  else soft_threshold_nonneg(nvec, vec, eta);
}

// TSV

// NOTE: the image array xvec/dvec is laid out flat as index = Ny*i + j for
// i in [0,Nx), j in [0,Ny) -- i.e. Nx rows of length Ny -- matching the
// convention used throughout the FFT/NUFFT machinery (see E4 in
// mfista_nufft_lib.cpp and the FFTW r2c row length Ny_h derivation in
// mfista_fft_lib.cpp). TSV/d_TSV must iterate with the same convention so
// that the regularization gradient lines up with the same pixels as the
// data-fidelity gradient; previously this used a row length of Nx instead
// of Ny, which is only equivalent when Nx==Ny.
double TSV(int Nx, int Ny, VectorXd &xvec, VectorXd &buf_diff)
{
  int i;
  double tsv = 0;

  for(i = 0; i < Nx-1; i++){
    buf_diff = xvec.segment(Ny*i,Ny-1)-xvec.segment(Ny*i+1,Ny-1);
    tsv += buf_diff.squaredNorm();
    buf_diff = xvec.segment(Ny*i,Ny-1)-xvec.segment(Ny*(i+1),Ny-1);
    tsv += buf_diff.squaredNorm();
    tsv += pow((xvec(Ny*i+(Ny-1))-xvec(Ny*(i+1)+(Ny-1))),2.0);
  }

  buf_diff = xvec.segment(Ny*(Nx-1),Ny-1)-xvec.segment(Ny*(Nx-1)+1,Ny-1);
  tsv += buf_diff.squaredNorm();

  return(tsv);
}

void d_TSV(VectorXd &dvec, int Nx, int Ny, VectorXd &xvec)
{
  int i;

  dvec = VectorXd::Zero(Nx*Ny);

  #ifdef _OPENMP
  #pragma omp parallel
  {
  #pragma omp for
  for(i = 0; i < Nx; i++){
    dvec.segment(Ny*i,(Ny-1))
      += 2*(xvec.segment(Ny*i,(Ny-1))-xvec.segment(Ny*i+1,(Ny-1)));
    dvec.segment(Ny*i+1,(Ny-1))
      += 2*(xvec.segment(Ny*i+1,(Ny-1))-xvec.segment(Ny*i,(Ny-1)));
  }

  #pragma omp for
  for(i = 0; i < Nx-1; i++){
    dvec.segment(Ny*i,Ny)
      += 2*(xvec.segment(Ny*i,Ny)-xvec.segment(Ny*(i+1),Ny));
  }
  #pragma omp for
  for(i = 0; i < Nx-1; i++){
    dvec.segment(Ny*(i+1),Ny)
      += 2*(xvec.segment(Ny*(i+1),Ny)-xvec.segment(Ny*i,Ny));
  }
  }
  #else
  for(i = 0; i < Nx; i++){
    dvec.segment(Ny*i,(Ny-1))
      += 2*(xvec.segment(Ny*i,(Ny-1))-xvec.segment(Ny*i+1,(Ny-1)));
    dvec.segment(Ny*i+1,(Ny-1))
      += 2*(xvec.segment(Ny*i+1,(Ny-1))-xvec.segment(Ny*i,(Ny-1)));
  }

  for(i = 0; i < Nx-1; i++){
    dvec.segment(Ny*i,Ny)
      += 2*(xvec.segment(Ny*i,Ny)-xvec.segment(Ny*(i+1),Ny));
    dvec.segment(Ny*(i+1),Ny)
      += 2*(xvec.segment(Ny*(i+1),Ny)-xvec.segment(Ny*i,Ny));
  }
  #endif
}

// utility for time measurement

void get_current_time(struct timespec *t) {
#ifdef __APPLE__
  struct timeval tv;
  struct timezone tz;
  int status = gettimeofday(&tv, &tz);
  if(status == 0) {
    t->tv_sec = tv.tv_sec;
    t->tv_nsec = tv.tv_usec * 1000; /* microsec -> nanosec */
  } else {
    t->tv_sec = 0.0;
    t->tv_nsec = 0.0;
  }
#else
  clock_gettime(CLOCK_MONOTONIC, t);
#endif
}


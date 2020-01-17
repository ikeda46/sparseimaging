#include "mfista.hpp"

#include <time.h>

#ifdef __APPLE__
#include <sys/time.h>
#endif

// mfista_tools

double calc_Q_part(VectorXd &xvec1, VectorXd &xvec2,
		   double c, VectorXd &AyAz, VectorXd &buf_vec)
{
  double term1, term2;

  // x1 - x2
  buf_vec = xvec1 - xvec2;
  // (x1 - x2)'A'(y - A x2)
  term1 = buf_vec.dot(AyAz);
  // (x1 - x2)'(x1 - x2)
  term2 = buf_vec.squaredNorm();

  return(-term1+c*term2/2);
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
  if(box_flag == 1) nvec = vec.array()*box.array();
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

double TSV(int Nx, int Ny, VectorXd &xvec, VectorXd &buf_diff)
{
  int j;
  double tsv = 0;

  for(j = 0; j < Ny-1; j++){
    buf_diff = xvec.segment(Nx*j,Nx-1)-xvec.segment(Nx*j+1,Nx-1);
    tsv += buf_diff.squaredNorm();
    buf_diff = xvec.segment(Nx*j,Nx-1)-xvec.segment(Nx*(j+1),Nx-1);
    tsv += buf_diff.squaredNorm();
    tsv += pow((xvec(Nx*j+(Nx-1))-xvec(Nx*(j+1)+(Nx-1))),2.0);
  }

  buf_diff = xvec.segment(Nx*(Ny-1),Nx-1)-xvec.segment(Nx*(Ny-1)+1,Nx-1);
  tsv += buf_diff.squaredNorm();

  return(tsv);
}

void d_TSV(VectorXd &dvec, int Nx, int Ny, VectorXd &xvec)
{
  int j;

  dvec = VectorXd::Zero(Nx*Ny);

  for(j = 0; j < Ny; j++){
    dvec.segment(Nx*j,(Nx-1))
      += 2*(xvec.segment(Nx*j,(Nx-1))-xvec.segment(Nx*j+1,(Nx-1)));
    dvec.segment(Nx*j+1,(Nx-1))
      += 2*(xvec.segment(Nx*j+1,(Nx-1))-xvec.segment(Nx*j,(Nx-1)));
  }
  
  for(j = 0; j < Ny-1; j++){
    dvec.segment(Nx*j,Nx)
      += 2*(xvec.segment(Nx*j,Nx)-xvec.segment(Nx*(j+1),Nx));
    dvec.segment(Nx*(j+1),Nx)
      += 2*(xvec.segment(Nx*(j+1),Nx)-xvec.segment(Nx*j,Nx));
  }
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


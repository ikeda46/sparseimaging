/* 
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista_lib.c'. An optimization algorithm for imaging
   of interferometry. The idea of the algorithm was from the following
   two papers,

   Beck and Teboulle (2009) SIAM J. Imaging Sciences, 
   Beck and Teboulle (2009) IEEE trans. on Image Processing 


   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/ 
#include "mfista.h" 

void fftw_half2full(int NX, int NY, fftw_complex *FT_h, doublecomplex *FT)
{
  int i, j, NX_h, NY_h;

  NX_h = floor(NX/2)+1;
  NY_h = floor(NY/2)+1;

  for(i=0;i<NX;++i)
    for(j=0;j<NY_h;++j){
      FT[NY*i+j].r = creal(FT_h[NY_h*i+j]);
      FT[NY*i+j].i = cimag(FT_h[NY_h*i+j]);
    }

  for(i=1;i<NX_h-1;++i)
    for(j=1;j<NY_h-1;++j){
      FT[NY*(NX-i)+(NY-j)].r =   creal(FT_h[NY_h*i+j]);
      FT[NY*(NX-i)+(NY-j)].i = - cimag(FT_h[NY_h*i+j]);
    }

  for(i=NX_h;i<NX;++i)
    for(j=1;j<NY_h-1;++j){
      FT[NY*(NX-i)+(NY-j)].r =   creal(FT_h[NY_h*i+j]);
      FT[NY*(NX-i)+(NY-j)].i = - cimag(FT_h[NY_h*i+j]);
    }

  for(j=1;j<NY_h-1;++j){
    FT[(NY-j)].r             =   creal(FT_h[j]);
    FT[(NY-j)].i             = - cimag(FT_h[j]);
    FT[NY*(NX_h-1)+(NY-j)].r =   creal(FT_h[NY_h*(NX_h-1)+j]);
    FT[NY*(NX_h-1)+(NY-j)].i = - cimag(FT_h[NY_h*(NX_h-1)+j]);
  }
}

void fftw_full2half(int NX, int NY, doublecomplex *FT, fftw_complex *FT_h)
{
  int i, j, NY_h;

  NY_h = floor(NY/2)+1;

  for(i=0;i<NX;++i)
    for(j=0;j<NY_h;++j)
      FT_h[NY_h*i+j] = (FT[NY*i+j].r) + (FT[NY*i+j].i)*I;
}

void calc_yAx_fftw(int NX, int NY,
		  fftw_complex *y_fft_h, double *mask, 
		  fftw_complex *yAx_h)
{
  int i, NY_h;
  double sqrtNN,tmp;

  NY_h   = floor(NY/2)+1;
  sqrtNN = sqrt(NX*NY);

  tmp = 0;
  for(i=0;i<NX*NY_h;++i){
    tmp += mask[i]*mask[i];
    yAx_h[i] = y_fft_h[i]-mask[i]*yAx_h[i]/sqrtNN;
  }
}

double calc_F_TSV_part_fftw(int *N, int NX, int NY,
			    fftw_complex *yf_h,
			    double *mask_h,
			    double lambda_tsv,
			    fftw_plan *fftwplan,
			    double *xvec,
			    fftw_complex *yAx_fh,
			    double *x4f,
			    doublecomplex *yAx_f)
{
  double term1, term2;
  doublecomplex result;
  int inc = 1;

  dcopy_(N, xvec, &inc, x4f, &inc);
  fftw_execute(*fftwplan);
  calc_yAx_fftw(NX, NY, yf_h, mask_h, yAx_fh);
  fftw_half2full(NX, NY, yAx_fh, yAx_f);

  result = zdotc_(N, yAx_f, &inc, yAx_f, &inc);

  term1 = result.r;
  term2 = TSV(NX, NY, xvec);

  return(term1/4+lambda_tsv*term2);
}

void dF_dx_fftw(int *N, int NX, int NY,
		fftw_complex *yAx_fh,
		double *mask_h, double *xvec, 
		double lambda_tsv,
		fftw_plan *ifftwplan,
		double *x4f, double *dFdx)
{
  int i, inc = 1;
  double sqNN = sqrt(*N), alpha = 1, beta= -lambda_tsv;

  for(i=0;i<NX*floor(NY/2+1);++i) yAx_fh[i] *= (mask_h[i]/(2*sqNN));

  fftw_execute(*ifftwplan);

  d_TSV(NX, NY, xvec, dFdx);

  dscal_(N, &beta, dFdx, &inc);

  daxpy_(N, &alpha, x4f, &inc, dFdx, &inc);
}

void mfista_L1_TSV_core_fftw(doublecomplex *yf, double *mask_h,
			     int M, int *N, int NX, int NY,
			     double lambda_l1, double lambda_tsv, double cinit,
			     double *xvec, int nonneg_flag, 
			     struct RESULT *mfista_result)
{
  void (*soft_th)(double *vector, int length, double eta, double *newvec);
  int i, iter, inc = 1, NY_h = (floor(NY/2)+1);
  double *xtmp, *xnew, *zvec, *dfdx, *x4f,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, costtmp, *cost,
    mu=1, munew, alpha = 1;
  fftw_complex *yf_h, *yAx_fh;
  fftw_plan fftwplan, ifftwplan;
  doublecomplex *yAx_f;

  printf("computing image with MFISTA.\n");

  /* malloc */

  cost  = alloc_vector(MAXITER);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  zvec  = alloc_vector(*N);
  x4f   = alloc_vector(*N);

  /* fftw malloc */

  yf_h   = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_f  = malloc((*N)*sizeof(doublecomplex));

  /* preparation for fftw */

  fftw_full2half(NX, NY, yf, yf_h);

  fftwplan  = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, FFTW_MEASURE );
  ifftwplan = fftw_plan_dft_c2r_2d( NX, NY, yAx_fh, x4f, FFTW_MEASURE );

  if(nonneg_flag == 0)
    soft_th=soft_threshold;
  else if(nonneg_flag == 1)
    soft_th=soft_threshold_nonneg;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return;
  }

  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  costtmp = calc_F_TSV_part_fftw(N, NX, NY, yf_h, mask_h, lambda_tsv,
				&fftwplan, xvec, yAx_fh, x4f, yAx_f);

  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    Qcore = calc_F_TSV_part_fftw(N, NX, NY, yf_h, mask_h, lambda_tsv,
				&fftwplan, zvec, yAx_fh, x4f, yAx_f);

    dF_dx_fftw(N, NX, NY, yAx_fh, mask_h, zvec, lambda_tsv,
	      &ifftwplan, x4f, dfdx);

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);
      soft_th(xtmp, *N, lambda_l1/c, xnew);
      Fval = calc_F_TSV_part_fftw(N, NX, NY, yf_h, mask_h, lambda_tsv,
				 &fftwplan, xnew, yAx_fh, x4f, yAx_f);
      Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);

    Fval += lambda_l1*l1cost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(N, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(N, xnew, &inc, xvec, &inc);
	    
    }	
    else{
      dcopy_(N, xvec, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(N, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(N, &tmpa, xnew, &inc, zvec, &inc);

      if((iter>1) && (dasum_(N, xvec, &inc) == 0)) break;
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

    mu = munew;
  }
  if(iter == MAXITER){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  mfista_result->M = M/2;
  mfista_result->N = (*N);
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_tsv = lambda_tsv;
  mfista_result->lambda_tv = 0;

  mfista_result->sq_error = calc_F_TSV_part_fftw(N, NX, NY, yf_h, mask_h, 0,
						&fftwplan, xvec, yAx_fh, x4f, yAx_f);

  mfista_result->mean_sq_error = mfista_result->sq_error/((double)M);

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < (*N);i++){
    tmpa = fabs(xvec[i]);
    if(tmpa > 0){
      mfista_result->l1cost += tmpa;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->tsvcost = TSV(NX, NY, xvec);
  mfista_result->tvcost = 0;
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda_l1*(mfista_result->l1cost)
    + lambda_tsv*(mfista_result->tsvcost);

  /* free */
  
  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);

  fftw_free(yf_h);

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);
}

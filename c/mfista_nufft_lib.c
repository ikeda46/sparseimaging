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

#include <assert.h>

#define NU_SIGN -1
#define MSP 12

#define RESTRICT restrict
#define ASSUME_ALIGNED(p, a) __builtin_assume_aligned((p), (a))

/* nufft */

int idx_fftw(int m, int Mr)
{
  if(m >= 0 && m < Mr/2)
    return(m);
  else if(m >= -Mr/2)
    return(m+Mr);
  else
    return(2*Mr);
}

int m2mr(int id, int N)
{
  if(id < N/2)
    return(id + 3*N/2);
  else
    return(id - N/2);
}

void preNUFFT(double * RESTRICT E1, double * RESTRICT E2x, double * RESTRICT E2y, double * RESTRICT E3x, double * RESTRICT E3y,
	      double * RESTRICT E4mat, int * RESTRICT mx, int * RESTRICT my,
	      double * RESTRICT u, double * RESTRICT v, int M, int Nx, int Ny)
{
  int i, j, k, l;
  double taux, tauy, coeff, xix, xiy, Mrx, Mry,
    tmpx, tmpy, pi, powj, tmpi, tmpj;

  double *E1_a = ASSUME_ALIGNED(E1, ALIGNMENT); 
  double *E2x_a = ASSUME_ALIGNED(E2x, ALIGNMENT); 
  double *E2y_a = ASSUME_ALIGNED(E2y, ALIGNMENT); 
  double *E3x_a = ASSUME_ALIGNED(E3x, ALIGNMENT); 
  double *E3y_a = ASSUME_ALIGNED(E3y, ALIGNMENT); 
  double *E4mat_a = ASSUME_ALIGNED(E4mat, ALIGNMENT); 
  double *u_a = ASSUME_ALIGNED(u, ALIGNMENT); 
  double *v_a = ASSUME_ALIGNED(v, ALIGNMENT); 
  int *mx_a = ASSUME_ALIGNED(mx, ALIGNMENT); 
  int *my_a = ASSUME_ALIGNED(my, ALIGNMENT); 

  pi = M_PI;

  Mrx = (double) 2*Nx;
  Mry = (double) 2*Ny;

  taux = 12/((double)(Nx*Nx));
  tauy = 12/((double)(Ny*Ny));

  coeff = pi/sqrt(taux*tauy);

  tmpx = pow((pi/Mrx),2.0)/taux;
  tmpy = pow((pi/Mry),2.0)/tauy;

  for(l = 0; l <= MSP; ++l){
    E3x_a[l] = exp(-tmpx*((double)(l*l)));
    E3y_a[l] = exp(-tmpy*((double)(l*l)));
  }

  for(i = 0 ; i < Nx; ++i){

    tmpi = (double)(i-Nx/2);
    tmpx = taux*tmpi*tmpi;

    for(j = 0; j< Ny; ++j){
      tmpj = (double)(j-Nx/2);
      tmpy = tauy*tmpj*tmpj;
      E4mat_a[Ny*i + j] = coeff*exp(tmpx + tmpy);
    }
  }

  for(k = 0; k < M; ++k){

    /* mx, my, xix, xiy */

    tmpx = round(u_a[k]*Mrx/(2*pi));
    tmpy = round(v_a[k]*Mry/(2*pi));
    
    mx_a[k] = (int)tmpx;
    my_a[k] = (int)tmpy;

    xix = (2*pi*tmpx)/Mrx;
    xiy = (2*pi*tmpy)/Mry;

    /* E1 */

    tmpx = -pow((u_a[k]-xix),2.0)/(4*taux);
    tmpy = -pow((v_a[k]-xiy),2.0)/(4*tauy);

    E1_a[k] = exp( tmpx + tmpy );

    /* E2 */

    tmpx = pi*(u_a[k]-xix)/(Mrx*taux);
    tmpy = pi*(v_a[k]-xiy)/(Mry*tauy);
    
    powj = (double)(-MSP+1);

    for(j = 0; j < 2*MSP; ++j){
      E2x_a[2*MSP*k + j] = exp(powj*tmpx);
      E2y_a[2*MSP*k + j] = exp(powj*tmpy);
      powj += 1.0;
    }
  }
}

void NUFFT2d1(double * RESTRICT Xout,
	      int M, int Nx, int Ny,
	      double * RESTRICT E1, double * RESTRICT E2x, double * RESTRICT E2y, double * RESTRICT E3x, double * RESTRICT E3y,
	      double * RESTRICT E4mat, int * RESTRICT mx, int * RESTRICT my,
	      fftw_complex * RESTRICT in, double * RESTRICT out, fftw_plan *fftwplan_c2r,
	      double * RESTRICT Fin_r, double * RESTRICT Fin_i)
{
  int j, k, lx, ly, Mrx, Mry, idx, idy, Mh;
  double v0_r, v0_i, vy_r, vy_i, MM, tmp;

  double *Fin_r_a = ASSUME_ALIGNED(Fin_r, ALIGNMENT); 
  double *Fin_i_a = ASSUME_ALIGNED(Fin_i, ALIGNMENT); 
  double *E1_a = ASSUME_ALIGNED(E1, ALIGNMENT); 
  double *E2x_a = ASSUME_ALIGNED(E2x, ALIGNMENT); 
  double *E2y_a = ASSUME_ALIGNED(E2y, ALIGNMENT); 
  double *E3x_a = ASSUME_ALIGNED(E3x, ALIGNMENT); 
  double *E3y_a = ASSUME_ALIGNED(E3y, ALIGNMENT); 
  double *E4mat_a = ASSUME_ALIGNED(E4mat, ALIGNMENT); 
  fftw_complex *in_a = ASSUME_ALIGNED(in, ALIGNMENT); 
  double *Xout_a = ASSUME_ALIGNED(Xout, ALIGNMENT); 
  double *out_a = ASSUME_ALIGNED(out, ALIGNMENT); 
  int *mx_a = ASSUME_ALIGNED(mx, ALIGNMENT); 
  int *my_a = ASSUME_ALIGNED(my, ALIGNMENT); 

  Mrx = 2*Nx;
  Mry = 2*Ny;
  Mh  = Ny+1;
  MM  = (double)(Mrx*Mry);
  
  for(j = 0; j < Mrx*Mh; ++j) in_a[j] = 0;

  for(k = 0; k<M; ++k){

    v0_r = Fin_r_a[k]*E1_a[k];
    v0_i = (NU_SIGN)*Fin_i_a[k]*E1_a[k];

    /* for original */
    
    for(ly = (-MSP+1); ly <= MSP; ++ly){

      tmp = E2y_a[2*MSP*k + (ly+MSP-1)]*E3y_a[abs(ly)];
      
      vy_r = v0_r*tmp;
      vy_i = v0_i*tmp;

      idy = idx_fftw(my_a[k]+ly,Mry);

      if(idy < (Ny+1))
	for(lx = (-MSP+1); lx <= MSP; ++lx){
	  idx = idx_fftw(mx_a[k]+lx,Mrx);
	  in_a[idx*Mh+idy] += (vy_r + I*vy_i)*E2x_a[2*MSP*k+(lx+MSP-1)]*E3x_a[abs(lx)]/2;
	}

      idy = idx_fftw(-(my_a[k]+ly),Mry);

      if(idy < (Ny+1))
	for(lx = (-MSP+1); lx <= MSP; ++lx){
	  idx = idx_fftw(-(mx_a[k]+lx),Mrx);
	  in_a[idx*Mh+idy] += (vy_r - I*vy_i)*E2x_a[2*MSP*k+(lx+MSP-1)]*E3x_a[abs(lx)]/2;
	}

    }
  }

  fftw_execute(*fftwplan_c2r);

  for(k = 0; k < Nx; ++k){

    idx = m2mr(k,Nx);
    for(j = 0; j < Ny; ++j){

      idy = m2mr(j,Ny);
      Xout_a[k*Ny + j] = out_a[idx*Mry + idy]*E4mat_a[k*Ny + j]/MM;
    }
  }
  
}

void NUFFT2d2(double * RESTRICT Fout_r, double * RESTRICT Fout_i,
	      int M, int Nx, int Ny,
	      double * RESTRICT E1, double * RESTRICT E2x, double * RESTRICT E2y, double * RESTRICT E3x, double * RESTRICT E3y,
	      double * RESTRICT E4mat, int * RESTRICT mx, int * RESTRICT my,
	      double * RESTRICT in, fftw_complex * RESTRICT out, fftw_plan *fftwplan_r2c,
	      double * RESTRICT Xin)
{
  int i, j, k, lx, ly, Mrx, Mry, Mh, idx, idy;
  double f_r, tmp, MM;
  double accr, acci;

  double *Fout_r_a = ASSUME_ALIGNED(Fout_r, ALIGNMENT); 
  double *Fout_i_a = ASSUME_ALIGNED(Fout_i, ALIGNMENT); 
  double *E1_a = ASSUME_ALIGNED(E1, ALIGNMENT); 
  double *E2x_a = ASSUME_ALIGNED(E2x, ALIGNMENT); 
  double *E2y_a = ASSUME_ALIGNED(E2y, ALIGNMENT); 
  double *E3x_a = ASSUME_ALIGNED(E3x, ALIGNMENT); 
  double *E3y_a = ASSUME_ALIGNED(E3y, ALIGNMENT); 
  double *E4mat_a = ASSUME_ALIGNED(E4mat, ALIGNMENT); 
  double *in_a = ASSUME_ALIGNED(in, ALIGNMENT); 
  double *Xin_a = ASSUME_ALIGNED(Xin, ALIGNMENT); 
  fftw_complex *out_a = ASSUME_ALIGNED(out, ALIGNMENT); 
  int *mx_a = ASSUME_ALIGNED(mx, ALIGNMENT); 
  int *my_a = ASSUME_ALIGNED(my, ALIGNMENT); 

  Mrx = 2*Nx;
  Mry = 2*Ny;
  Mh  = Ny+1;
  MM = (double)Mrx*Mry;

  for( i = 0; i < Nx; ++i){
    idx = m2mr(i,Nx);
    for( j = 0; j < Ny; ++j){
      idy = m2mr(j,Ny);
      in_a[idx*Mry + idy] = Xin_a[i*Ny + j]*E4mat_a[i*Ny + j];
    }
  }

  fftw_execute(*fftwplan_r2c);

  for(k = 0; k < M; ++k) for(ly = (-MSP+1); ly <= MSP; ++ly){
	
      f_r = E1_a[k]*E2y_a[2*MSP*k + (ly+MSP-1)]*E3y_a[abs(ly)];

      idy = idx_fftw((my_a[k]+ly),Mry);
      
      accr = 0;
      acci = 0;

      if(idy < (Ny+1)){
	for(lx = (-MSP+1); lx <= MSP; ++lx){

	  idx = idx_fftw((mx_a[k]+lx),Mrx);
	  tmp = E2x_a[2*MSP*k + (lx+MSP-1)]*E3x_a[abs(lx)];

	  accr +=           creal(out_a[idx*Mh + idy])*tmp;
	  acci += (NU_SIGN)*cimag(out_a[idx*Mh + idy])*tmp;
	}
      }

      else{
	idy = idx_fftw(-(my_a[k]+ly),Mry);
	
	if(idy < (Ny+1)) for(lx = (-MSP+1); lx <= MSP; ++lx){
	    
	    idx = idx_fftw(-(mx_a[k]+lx),Mrx);
	    tmp = E2x_a[2*MSP*k + (lx+MSP-1)]*E3x_a[abs(lx)];
	    
	    accr +=            creal(out_a[idx*Mh + idy])*tmp;
	    acci += -(NU_SIGN)*cimag(out_a[idx*Mh + idy])*tmp;
	  }
      }

      Fout_r_a[k] += f_r * accr / MM;
      Fout_i_a[k] += f_r * acci / MM;
    }

}

void calc_yAx_nufft(double * RESTRICT yAx_r, double * RESTRICT yAx_i,
		    int M, double * RESTRICT vis_r, double * RESTRICT vis_i, double * RESTRICT weight)
{
  int i;

  double *yAx_r_a = ASSUME_ALIGNED(yAx_r, ALIGNMENT); 
  double *yAx_i_a = ASSUME_ALIGNED(yAx_i, ALIGNMENT); 
  double *vis_r_a = ASSUME_ALIGNED(vis_r, ALIGNMENT); 
  double *vis_i_a = ASSUME_ALIGNED(vis_i, ALIGNMENT); 
  double *weight_a = ASSUME_ALIGNED(weight, ALIGNMENT); 

  /* main */

  for(i = 0; i < M; ++i){
    yAx_r_a[i] = (vis_r_a[i] - yAx_r_a[i])*weight_a[i];
    yAx_i_a[i] = (vis_i_a[i] - yAx_i_a[i])*weight_a[i];
  }
}

double calc_F_part_nufft(double * RESTRICT yAx_r, double * RESTRICT yAx_i,
			 int M, int Nx, int Ny,
			 double * RESTRICT E1, double * RESTRICT E2x, double * RESTRICT E2y,
			 double * RESTRICT E3x, double * RESTRICT E3y, double * RESTRICT E4mat, int * RESTRICT mx, int * RESTRICT my,
			 double * RESTRICT in_r, fftw_complex * RESTRICT out_c, double * RESTRICT dzeros, fftw_plan *fftwplan_r2c,
			 double * RESTRICT vis_r, double * RESTRICT vis_i, double * RESTRICT weight, double * RESTRICT xvec)
{
  int inc = 1, Ml = M, MM = 4*Nx*Ny;
  double chisq;
  int i;

  double *yAx_r_a = ASSUME_ALIGNED(yAx_r, ALIGNMENT); 
  double *yAx_i_a = ASSUME_ALIGNED(yAx_i, ALIGNMENT); 
  double *in_r_a = ASSUME_ALIGNED(in_r, ALIGNMENT); 

  /* initializaton for nufft */
  /* replaced dcopy_ with simple loop */
  for (i = 0; i < MM; ++i) {
    in_r_a[i] = 0;
  }
  for (i = 0; i < Ml; ++i) {
    yAx_r_a[i] = 0;
    yAx_i_a[i] = 0;
  }

  /* nufft */
  
  NUFFT2d2(yAx_r, yAx_i, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
	   in_r, out_c, fftwplan_r2c, xvec);

  calc_yAx_nufft(yAx_r, yAx_i, M, vis_r, vis_i, weight);

  chisq  = ddot_(&Ml, yAx_r, &inc, yAx_r, &inc);
  chisq += ddot_(&Ml, yAx_i, &inc, yAx_i, &inc);

  return(chisq/2);
}

void dF_dx_nufft(double * RESTRICT dFdx,
		 int M, int Nx, int Ny,
		 double * RESTRICT E1, double * RESTRICT E2x, double * RESTRICT E2y, double * RESTRICT E3x, double * RESTRICT E3y,
		 double * RESTRICT E4mat, int * RESTRICT mx, int * RESTRICT my,
		 fftw_complex * RESTRICT in_c, double * RESTRICT out_r, fftw_plan *fftwplan_c2r, double * RESTRICT weight,
		 double * RESTRICT yAx_r, double * RESTRICT yAx_i)
{
  int i;

  double *yAx_r_a = ASSUME_ALIGNED(yAx_r, ALIGNMENT); 
  double *yAx_i_a = ASSUME_ALIGNED(yAx_i, ALIGNMENT); 
  double *weight_a = ASSUME_ALIGNED(weight, ALIGNMENT); 

  for(i = 0; i < M; ++i){
    yAx_r_a[i] *= weight_a[i];
    yAx_i_a[i] *= weight_a[i];
  }

  NUFFT2d1(dFdx, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat,
	   mx, my, in_c, out_r, fftwplan_c2r, yAx_r, yAx_i);

}

/* TV */

int mfista_L1_TV_core_nufft(double *xout,
			    int M, int Nx, int Ny,
			    double *u_dx, double *v_dy,
			    int maxiter, double eps,
			    double *vis_r, double *vis_i, double *vis_std,
			    double lambda_l1, double lambda_tv,
			    double *cinit, double *xinit, 
			    int nonneg_flag, int box_flag, float *cl_box)
{
  int NN = Nx*Ny, MMh = 2*Nx*(Ny+1), i, j, iter, inc = 1, *mx, *my;
  double Qcore, Fval, Qval, c, costtmp, mu=1, munew, alpha = 1, tmpa, l1cost, tvcost,
    *weight, *yAx_r, *yAx_i, *E1, *E2x, *E2y, *E3x, *E3y, *E4mat, *rvec, *dzeros,
    *zvec, *xtmp, *xnew, *dfdx, *rmat, *smat, *npmat, *nqmat, *pmat, *qmat, *cost;
  fftw_complex *cvec;
  fftw_plan fftwplan_c2r, fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  /* set parameters */
  
  printf("computing image with MFISTA with NUFFT.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* prepare for nufft */

  E1  = alloc_vector(M);
  E2x = alloc_vector(2*MSP*M);
  E2y = alloc_vector(2*MSP*M);
  E3x = alloc_vector(MSP+1);
  E3y = alloc_vector(MSP+1);
  E4mat = alloc_vector(NN);
  mx  = alloc_int_vector(M);
  my  = alloc_int_vector(M);

  preNUFFT(E1, E2x, E2y, E3x, E3y, E4mat, mx, my, u_dx, v_dy, M, Nx, Ny);

  /* for fftw */
  
  //cvec  = (fftw_complex*) fftw_malloc(MMh*sizeof(fftw_complex));
  assert(sizeof(fftw_complex) == 2 * sizeof(double));
  cvec  = (fftw_complex*) alloc_vector(2*MMh);
  rvec  = alloc_vector(4*NN);

  // do not allocate dzeros for memory efficiency
  dzeros = NULL;
  
  fftwplan_c2r = fftw_plan_dft_c2r_2d(2*Nx,2*Ny, cvec, rvec, fftw_plan_flag);
  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvec, cvec, fftw_plan_flag);

  /* allocate variables for NUFFT */

  weight = alloc_vector(M);
  
  yAx_r = alloc_vector(M);
  yAx_i = alloc_vector(M);

  /* allocate variables */ 

  cost  = alloc_vector(maxiter);
  dfdx  = alloc_vector(NN);
  xnew  = alloc_vector(NN);
  xtmp  = alloc_vector(NN);
  zvec  = alloc_vector(NN);

  /* preparation for TV */

  pmat = alloc_matrix(Nx-1,Ny);
  qmat = alloc_matrix(Nx,Ny-1);

  npmat = alloc_matrix(Nx-1,Ny);
  nqmat = alloc_matrix(Nx,Ny-1);

  rmat = alloc_matrix(Nx-1,Ny);
  smat = alloc_matrix(Nx,Ny-1);

  /* initialize xvec */
  
  dcopy_(&NN, xinit, &inc, xout, &inc);
  dcopy_(&NN, xinit, &inc, zvec, &inc);

  c = *cinit;

  for(i = 0; i < M; ++i) weight[i] = 1/vis_std[i];

  /* main */

  costtmp = calc_F_part_nufft(yAx_r, yAx_i, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			      rvec, cvec, dzeros, &fftwplan_r2c,
			      vis_r, vis_i, weight, xout);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  tvcost = TV(Nx, Ny, xout);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 10) == 0) printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_nufft(yAx_r, yAx_i, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			      rvec, cvec, dzeros, &fftwplan_r2c,
			      vis_r, vis_i, weight, zvec);

    dF_dx_nufft(dfdx, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
		cvec, rvec, &fftwplan_c2r, weight, yAx_r, yAx_i);


    for(i = 0; i < maxiter; i++){

      if(nonneg_flag == 1){
	for(j = 0; j < NN; ++j) xtmp[j]=1;
	tmpa = -lambda_l1/c;
	dscal_(&NN, &tmpa, xtmp, &inc);
	tmpa = 1/c;
	daxpy_(&NN, &tmpa, dfdx, &inc, xtmp, &inc);
	daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);

	FGP_nonneg_box(&NN, Nx, Ny, xtmp, lambda_tv/c, FGPITER,
		       pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		       box_flag, cl_box);
      }
      else{
	dcopy_(&NN, dfdx, &inc, xtmp, &inc);
	tmpa = 1/c;
	dscal_(&NN, &tmpa, xtmp, &inc);
	daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);

	FGP_L1_box(&NN, Nx, Ny, xtmp, lambda_l1/c, lambda_tv/c, FGPITER,
		   pmat, qmat, rmat, smat, npmat, nqmat, xnew,
		   box_flag, cl_box);
      }

      Fval = calc_F_part_nufft(yAx_r, yAx_i,
			       M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			       rvec, cvec, dzeros, &fftwplan_r2c, vis_r, vis_i, weight, xnew);

      
      Qval = calc_Q_part(&NN, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1 + sqrt(1 + 4*mu*mu))/2;

    l1cost = dasum_(&NN, xnew, &inc);
    Fval += lambda_l1*l1cost;

    tvcost = TV(Nx, Ny, xnew);
    Fval += lambda_tv*tvcost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(&NN, xnew, &inc, xout, &inc);
    }	
    else{
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(&NN, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(&NN, xout, &inc) == 0)){
	printf("x becomes a 0 vector.\n");
	break;
      }
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])< eps )) break;

    mu = munew;
  }
  if(iter == maxiter){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  *cinit = c;

  /* free NUFFT */

  free(E1);
  free(E2x);
  free(E2y);
  free(E3x);
  free(E3y);
  free(E4mat);
  free(mx);
  free(my);

  free(cvec);
  free(rvec);

  fftw_destroy_plan(fftwplan_c2r);
  fftw_destroy_plan(fftwplan_r2c);

  free(weight);

  free(yAx_r);
  free(yAx_i);

  /* clear memory */

  free(npmat);
  free(nqmat);
  free(pmat);
  free(qmat);
  free(rmat);
  free(smat);

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);

  return(iter+1);
}

/* TSV */

int mfista_L1_TSV_core_nufft(double *xout,
			     int M, int Nx, int Ny, double *u_dx, double *v_dy,
			     int maxiter, double eps,
			     double *vis_r, double *vis_i, double *vis_std,
			     double lambda_l1, double lambda_tsv,
			     double *cinit, double *xinit, 
			     int nonneg_flag, int box_flag, float *cl_box)
{
  void (*soft_th_box)(double *vector, int length, double eta, double *newvec,
		      int box_flag, float *cl_box);
  int NN = Nx*Ny, MMh = 2*Nx*(Ny+1), i, iter, inc = 1, *mx, *my;
  double Qcore, Fval, Qval, c, cinv, tmpa, l1cost, tsvcost, costtmp, 
    mu=1, munew, alpha = 1, beta = -lambda_tsv,
    *weight, *yAx_r, *yAx_i, *E1, *E2x, *E2y, *E3x, *E3y, *E4mat, *rvec, *dzeros,
    *cost, *xtmp, *xnew, *zvec, *dfdx, *dtmp;
  fftw_complex *cvec;
  fftw_plan fftwplan_c2r, fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  /* set parameters */

  printf("computing image with MFISTA with NUFFT.\n");

  printf("stop if iter = %d, or Delta_cost < %e\n", maxiter, eps);

  /* prepare for nufft */

  E1  = alloc_vector(M);
  E2x = alloc_vector(2*MSP*M);
  E2y = alloc_vector(2*MSP*M);
  E3x = alloc_vector(MSP+1);
  E3y = alloc_vector(MSP+1);
  E4mat = alloc_vector(NN);
  mx  = alloc_int_vector(M);
  my  = alloc_int_vector(M);

  preNUFFT(E1, E2x, E2y, E3x, E3y, E4mat, mx, my, u_dx, v_dy, M, Nx, Ny);

  /* for fftw */
  
  cvec  = (fftw_complex*) fftw_malloc(MMh*sizeof(fftw_complex));
  rvec  = alloc_vector(4*NN);
  
  // do not allocate dzeros for memory efficiency
  dzeros = NULL;

  fftwplan_c2r = fftw_plan_dft_c2r_2d(2*Nx,2*Ny, cvec, rvec, fftw_plan_flag);
  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvec, cvec, fftw_plan_flag);

  /* allocate variables */

  cost   = alloc_vector(maxiter);
  dfdx   = alloc_vector(NN);
  xnew   = alloc_vector(NN);
  xtmp   = alloc_vector(NN);
  zvec   = alloc_vector(NN);
  dtmp   = alloc_vector(NN);

  /* allocate variables for NUFFT */

  weight = alloc_vector(M);
  
  yAx_r = alloc_vector(M);
  yAx_i = alloc_vector(M);
  
  /* initialization */

  if(nonneg_flag == 0)
    soft_th_box =soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box =soft_threshold_nonneg_box;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  dcopy_(&NN, xinit, &inc, xout, &inc);
  dcopy_(&NN, xinit, &inc, zvec, &inc);

  c = *cinit;

  for(i = 0; i < M; ++i) weight[i] = 1/vis_std[i];

  /* main */

  costtmp = calc_F_part_nufft(yAx_r, yAx_i,
			      M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			      rvec, cvec, dzeros, &fftwplan_r2c, vis_r, vis_i, weight, xout);

  l1cost = dasum_(&NN, xout, &inc);
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(Nx, Ny, xout);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost[iter] = costtmp;

    if((iter % 10) == 0)
      printf("%d cost = %f, c = %f \n",(iter+1), cost[iter], c);

    Qcore = calc_F_part_nufft(yAx_r, yAx_i,
			      M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			      rvec, cvec, dzeros, &fftwplan_r2c, vis_r, vis_i, weight, zvec);

    dF_dx_nufft(dfdx, M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
		cvec, rvec, &fftwplan_c2r, weight, yAx_r, yAx_i);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(Nx, Ny, zvec);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(Nx, Ny, zvec, dtmp);
      dscal_(&NN, &beta, dtmp, &inc);
      daxpy_(&NN, &alpha, dtmp, &inc, dfdx, &inc);
    }

    for( i = 0; i < maxiter; i++){
      dcopy_(&NN, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(&NN, &cinv, xtmp, &inc);
      daxpy_(&NN, &alpha, zvec, &inc, xtmp, &inc);
      soft_th_box(xtmp, NN, lambda_l1/c, xnew, box_flag, cl_box);

      Fval = calc_F_part_nufft(yAx_r, yAx_i,
			       M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			       rvec, cvec, dzeros, &fftwplan_r2c, vis_r, vis_i, weight, xnew);

      if( lambda_tsv > 0.0 ){
	tsvcost = TSV(Nx, Ny, xnew);
	Fval += lambda_tsv*tsvcost;
      }

      Qval = calc_Q_part(&NN, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;
      
      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(&NN, xnew, &inc);
    Fval += lambda_l1*l1cost;

    if(Fval < cost[iter]){

      costtmp = Fval;
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = (1-mu)/munew;
      dscal_(&NN, &tmpa, zvec, &inc);

      tmpa = 1+((mu-1)/munew);
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);
	
      dcopy_(&NN, xnew, &inc, xout, &inc);
	    
    }
    else{
      dcopy_(&NN, xout, &inc, zvec, &inc);

      tmpa = 1-(mu/munew);
      dscal_(&NN, &tmpa, zvec, &inc);
      
      tmpa = mu/munew;
      daxpy_(&NN, &tmpa, xnew, &inc, zvec, &inc);

      /* another stopping rule */
      if((iter>1) && (dasum_(&NN, xout, &inc) == 0)) break;
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])< eps )) break;

    mu = munew;
  }
  if(iter == maxiter){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  *cinit = c;

  /* free */
  
  free(E1);
  free(E2x);
  free(E2y);
  free(E3x);
  free(E3y);
  free(E4mat);
  free(mx);
  free(my);

  free(cvec);
  free(rvec);

  fftw_destroy_plan(fftwplan_c2r);
  fftw_destroy_plan(fftwplan_r2c);

  free(weight);

  free(yAx_r);
  free(yAx_i);

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);
  free(dtmp);

  return(iter+1);
}

/* results */

void calc_result_nufft(struct RESULT *mfista_result,
		       int M, int Nx, int Ny, double *u_dx, double *v_dy,
		       double *vis_r, double *vis_i, double *vis_std,
		       double lambda_l1, double lambda_tv, double lambda_tsv, 
		       double *xvec)
{
  int i, NN = Nx*Ny, *mx, *my;
  double tmp, *yAx_r, *yAx_i, *weight, *E1, *E2x, *E2y, *E3x, *E3y, *E4mat, *rvec, *dzeros;
  fftw_complex *cvec;
  fftw_plan fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  /* prepare for nufft */

  E1  = alloc_vector(M);
  E2x = alloc_vector(2*MSP*M);
  E2y = alloc_vector(2*MSP*M);
  E3x = alloc_vector(MSP+1);
  E3y = alloc_vector(MSP+1);
  E4mat = alloc_vector(NN);
  mx  = alloc_int_vector(M);
  my  = alloc_int_vector(M);

  preNUFFT(E1, E2x, E2y, E3x, E3y, E4mat, mx, my, u_dx, v_dy, M, Nx, Ny);

  /* for fftw */
  
  //cvec  = (fftw_complex*) fftw_malloc(2*Nx*(Ny+1)*sizeof(fftw_complex));
  assert(sizeof(fftw_complex) == 2 * sizeof(double));
  cvec = (fftw_complex*) alloc_vector(2*2*Nx*(Ny+1));
  rvec  = alloc_vector(4*NN);

  // do not allocate dzeros for memory efficiency
  dzeros = NULL;

  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvec, cvec, fftw_plan_flag);

  /* complex malloc */

  yAx_r     = alloc_vector(M);
  yAx_i     = alloc_vector(M);
  weight    = alloc_vector(M);

  for(i = 0; i < M; ++i) weight[i] = 1/vis_std[i];

  /* computing results */
  
  tmp = calc_F_part_nufft(yAx_r, yAx_i,
			  M, Nx, Ny, E1, E2x, E2y, E3x, E3y, E4mat, mx, my,
			  rvec, cvec, dzeros, &fftwplan_r2c, vis_r, vis_i, weight, xvec);

  /* saving results */

  mfista_result->sq_error = 2*tmp;

  mfista_result->M  = (int)(M/2);
  mfista_result->N  = NN;
  mfista_result->NX = Nx;
  mfista_result->NY = Ny;
	    
  mfista_result->lambda_l1  = lambda_l1;
  mfista_result->lambda_tv  = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;
  
  mfista_result->mean_sq_error = mfista_result->sq_error/((double)M);

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < NN;++i){
    tmp = fabs(xvec[i]);
    if(tmp > 0){
      mfista_result->l1cost += tmp;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->finalcost = (mfista_result->sq_error)/2;

  if(lambda_l1 > 0)
    mfista_result->finalcost += lambda_l1*(mfista_result->l1cost);

  if(lambda_tsv > 0){
    mfista_result->tsvcost = TSV(Nx, Ny, xvec);
    mfista_result->finalcost += lambda_tsv*(mfista_result->tsvcost);
  }
  else if (lambda_tv > 0){
    mfista_result->tvcost = TV(Nx, Ny, xvec);
    mfista_result->finalcost += lambda_tv*(mfista_result->tvcost);
  }

  /* free */

  free(E1);
  free(E2x);
  free(E2y);
  free(E3x);
  free(E3y);
  free(E4mat);
  free(mx);
  free(my);

  free(cvec);
  free(rvec);

  fftw_destroy_plan(fftwplan_r2c);

  free(yAx_r);
  free(yAx_i);
  free(weight);
}

/* main subroutine */

void mfista_imaging_core_nufft(double *u_dx, double *v_dy, 
			       double *vis_r, double *vis_i, double *vis_std,
			       int M, int Nx, int Ny, int maxiter, double eps,
			       double lambda_l1, double lambda_tv, double lambda_tsv,
			       double cinit, double *xinit, double *xout,
			       int nonneg_flag, int box_flag, float *cl_box,
			       struct RESULT *mfista_result)
{
  int iter = 0, Ml = M, inc = 1;
  double epsilon, s_t, e_t, c = cinit;
  struct timespec time_spec1, time_spec2;

  /* start main part */

  epsilon  = ddot_(&Ml, vis_r, &inc, vis_r, &inc);
  epsilon += ddot_(&Ml, vis_i, &inc, vis_i, &inc);
    
  epsilon *= eps/((double)M);

  get_current_time(&time_spec1);

  if( lambda_tv == 0 ){
    iter = mfista_L1_TSV_core_nufft(xout, M, Nx, Ny, u_dx, v_dy, maxiter, epsilon, vis_r, vis_i, vis_std,
				    lambda_l1, lambda_tsv, &c, xinit, nonneg_flag, box_flag, cl_box);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core_nufft(xout, M, Nx, Ny, u_dx, v_dy, maxiter, epsilon, vis_r, vis_i, vis_std,
				   lambda_l1, lambda_tv, &c, xinit, nonneg_flag, box_flag, cl_box);
  }
  else{
    printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
    return;
  }

  get_current_time(&time_spec2);

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER      = iter;
  mfista_result->nonneg    = nonneg_flag;
  mfista_result->Lip_const = c;
  mfista_result->maxiter   = maxiter;

  calc_result_nufft(mfista_result, M, Nx, Ny, u_dx, v_dy, vis_r, vis_i, vis_std,
		    lambda_l1, lambda_tv, lambda_tsv, xout);

}

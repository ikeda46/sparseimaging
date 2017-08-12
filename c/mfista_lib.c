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

/* subroutines for mfista*/

void dL_dx(int *M, int *N, double *yAx, double *Amatrix, double *dfdx)
{
  int inc = 1;
  double beta = 1, gamma = 0;

  /* A' (y - A x) */
  dgemv_("T", M, N, &beta, Amatrix, M, yAx, &inc, &gamma, dfdx, &inc);

}

int mfista_L1_core(double *yvec, double *Amat, int *M, int *N, 
		   double lambda, double cinit,
		   double *xvec, int nonneg_flag)
{
  double *ytmp, *dfdx, *zvec, *xtmp, *xnew,
    Qcore, Fval, Qval, c, cinv, costtmp, *cost, l1cost,
    mu=1, munew, alpha = 1, tmpa;
  int i, iter, inc = 1;
  void (*soft_th)(double *vector, int length, double eta, double *newvec);

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  cost  = alloc_vector(MAXITER);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  ytmp  = alloc_vector(*M);
  zvec  = alloc_vector(*N);

  /* defining soft_thresholding */
  
  if(nonneg_flag == 0)
    soft_th=soft_threshold;
  else if(nonneg_flag == 1)
    soft_th=soft_threshold_nonneg;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }
  
  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  calc_yAz(M, N, yvec, Amat, xvec, ytmp);
  costtmp = ddot_(M, ytmp, &inc, ytmp, &inc)/2;

  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda*l1cost;
  
  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %lf\n",(iter+1), cost[iter]);

    calc_yAz(M, N, yvec, Amat, zvec, ytmp);

    dL_dx(M, N, ytmp, Amat, dfdx);

    Qcore = ddot_(M, ytmp, &inc, ytmp, &inc)/2;

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc); 
      soft_th(xtmp, *N, lambda/c, xnew);

      calc_yAz(M, N, yvec, Amat, xnew, ytmp);
      Fval = ddot_(M, ytmp, &inc, ytmp, &inc)/2;
      
      Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);

    Fval += lambda*l1cost;

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

      /* another stopping rule */
      if((iter>1) && (dasum_(N, xvec, &inc) == 0)){
	  printf("x becomes a 0 vector.\n");
	  break;
	}
    }

    /* stopping rule start */

    if(iter>=MINITER){
      if((cost[iter-TD]-cost[iter])<EPS) break;
    }

    /* stopping rule end */

    mu = munew;
  }
  if(iter == MAXITER){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);
    
  printf("\n");

  /* clear memory */

  free(cost);
  free(dfdx);
  free(xtmp);
  free(xnew);
  free(ytmp);
  free(zvec);
  return(iter+1);
}

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

double calc_F(int *M, int *N,
	      double *yvec, double *Amatrix,
	      double *xvec, double lambda,
	      int *inc, double *buffvec)
{
  double term1, term2, alpha = -1, beta = 1;

  dcopy_(M, yvec, inc, buffvec, inc);

  dgemv_("N", M, N, &alpha, Amatrix, M, 
	 xvec, inc, &beta, buffvec, inc);

  term1 = ddot_(M, buffvec, inc, buffvec, inc);
  term2 = dasum_(N, xvec, inc);

  return(term1/2+lambda*term2);
}


void df_dx(int *M, int *N,
	   double *yvec, double *Amatrix, 
	   double *xvec, 
	   int *inc, double *buffyvec1, double *dfdx)
{
  double alpha = -1, beta = 1, gamma = 0;
  /* y - A x */
  dcopy_(M, yvec, inc, buffyvec1, inc);
  dgemv_("N", M, N, &alpha, Amatrix, M, 
	 xvec, inc, &beta, buffyvec1, inc);

  /* A' (y - A x) */
  dgemv_("T", M, N, &beta, Amatrix, M, 
	 buffyvec1, inc, &gamma, dfdx, inc);

}

double *compute_Hessian_L1(int *M,  int *indx_list,
			   double *Amat_s, int N_active)
{
  int i,j;
  double *Hessian, alpha = 1, beta = 0;

  printf("The size of Hessian is %d x %d. ",N_active,N_active);

  Hessian = alloc_matrix(N_active,N_active);

  for(i=0;i<N_active;i++)
    for(j=0;j<N_active;j++)
      Hessian[i*N_active+j]=0;

  dsyrk_("L", "T", &N_active, M, &alpha, Amat_s, M,
	 &beta, Hessian, &N_active);

  printf("Done.\n");
  return(Hessian);
}

double compute_LOOE_L1(int *M, int *N, double lambda1, 
		       double *yvec, double *Amat, double *xvec, double *yAx)
{
  double *Amat_s, *Hessian, LOOE;
  int    N_active, *indx_list;

  /* computing LOOE */

  indx_list = (int *)malloc(sizeof(int)*(*N));

  N_active = find_active_set(*N, xvec, indx_list);

  Amat_s = shrink_A(*M, *N, N_active, indx_list, Amat);

  printf("The number of active components is %d\n",N_active);

  printf("Computing Hessian matrix.\n");
  Hessian = compute_Hessian_L1(M, indx_list, Amat_s, N_active);

  printf("\n");
  LOOE = compute_LOOE_core(M, N_active, yvec, Amat, xvec, yAx, Amat_s, Hessian);

  printf("LOOE = %lg\n",LOOE);

  free(Amat_s);
  free(Hessian);
  free(indx_list);

  return(LOOE);
}


void mfista_L1_core(double *yvec, double *Amat, int *M, int *N, 
		    double lambda, double cinit,
		    double *xvec, int nonneg_flag, int looe_flag,
		    struct RESULT *mfista_result)
{
  double *ytmp, *dfdx, *zvec, *xtmp, *xnew, *yAx,
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
    return;
  }
  
  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  costtmp = calc_F(M, N, yvec, Amat, xvec, lambda, &inc, ytmp);

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %lf\n",(iter+1), cost[iter]);

    df_dx(M, N, yvec, Amat, zvec, &inc, ytmp, dfdx);

    Qcore = calc_F_part(M, N, yvec, Amat, zvec, &inc, ytmp);

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc); 
      soft_th(xtmp, *N, lambda/c, xnew);
      Fval = calc_F_part(M, N, yvec, Amat, xnew, &inc, ytmp);
      Qval = calc_Q_part(N, xnew, zvec, c, &inc, dfdx, xtmp);
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
    
  /* sending summary of results */

  printf("\n");

  mfista_result->M = (*M);
  mfista_result->N = (*N);
  mfista_result->NX = 0;
  mfista_result->NY = 0;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda;
  mfista_result->lambda_sqtv = 0;
  mfista_result->lambda_tv = 0;

  yAx  = alloc_vector(*M);
  calc_yAz(M, N, yvec, Amat, xvec, &inc, yAx);

  /* mean square error */
  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++){
    mfista_result->sq_error += yAx[i]*yAx[i];
  }

  /* average of mean square error */

  mfista_result->mean_sq_error = mfista_result->sq_error/((double)(*M));

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < (*N);i++){
    tmpa = fabs(xvec[i]);
    if(tmpa > 0){
      mfista_result->l1cost += tmpa;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->sqtvcost = 0;
  mfista_result->tvcost   = 0;
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda*(mfista_result->l1cost);

  /* computing LOOE */

  if(looe_flag == 1){
    mfista_result->looe = compute_LOOE_L1(M, N, lambda,
					  yvec, Amat, xvec, yAx);
    if(mfista_result->looe == -1){
      mfista_result->Hessian_positive = 0;
      mfista_result->looe = 0;
    }
    else{
      mfista_result->Hessian_positive = 1;
    }
  }
  else{
    mfista_result->looe = 0;
    mfista_result->Hessian_positive = -1;
  }
    
  free(yAx);

  /* clear memory */

  free(cost);
  free(dfdx);
  free(xtmp);
  free(xnew);
  free(ytmp);
  free(zvec);
}

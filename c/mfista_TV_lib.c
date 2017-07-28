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

/* subroutines for mfista_tv */

void Lx(int NX, int NY, double *xvec, double *p, double *q)
{
  int i, j;

  for(i = 0; i < NX-1; ++i)for(j = 0; j < NY; ++j)
      p[(NX-1)*j+i] = xvec[NX*j+i]-xvec[NX*j+i+1];

  for(i = 0; i < NX; ++i)for(j = 0; j < NY-1; ++j)
      q[NX*j+i] = xvec[NX*j+i]-xvec[NX*(j+1)+i];
}

void Lpq(int NX, int NY, double *pmat, double *qmat, double *xvec)
{
  int i, j;
  
  for(i = 0; i < NX*NY; ++i)
    xvec[i] = 0;
  
  for(i = 0; i < NX-1; ++i)for(j = 0; j < NY; ++j){
      xvec[NX*j+i]   += pmat[(NX-1)*j+i];
      xvec[NX*j+i+1] -= pmat[(NX-1)*j+i];
    }

  for(i = 0; i < NX; ++i)for(j = 0; j < NY-1; ++j){
      xvec[NX*j+i]     += qmat[NX*j+i];
      xvec[NX*(j+1)+i] -= qmat[NX*j+i];
    }
}

double TV(int NX, int NY, double *xvec)
{
  int i, j;
  double tv = 0;

  for(i = 0; i < NX-1; ++i) for(j = 0; j < NY-1; ++j)
      tv += sqrt(pow((xvec[NX*j+i]-xvec[NX*j+i+1]),2.0)
		 +pow((xvec[NX*j+i]-xvec[NX*(j+1)+i]),2.0));

  for(i = 0; i < NX-1; ++i)
    tv += fabs(xvec[NX*(NY-1)+i]- xvec[NX*(NY-1)+i+1]);

  for(j = 0; j < NY-1; ++j)
    tv += fabs(xvec[NX*j+NX-1]  - xvec[NX*(j+1)+NX-1]);

  return(tv);
}

void P_positive(int N, double *vec)
{
  int i;
  
  for(i = 0; i< N; ++i) if(vec[i]<0) vec[i] = 0;
}

void P_pqiso(int NX, int NY,
	     double *pmat, double *qmat,
	     double *rmat, double *smat)
{
  int i,j;
  double tmp;

  for(i = 0; i < NX-1; ++i) for(j = 0; j < NY-1; ++j){
      tmp = sqrt(pow(pmat[(NX-1)*j+i],2.0) + pow(qmat[NX*j+i],2.0));

      if(tmp<1) tmp = 1;
      
      rmat[(NX-1)*j+i] = pmat[(NX-1)*j+i]/tmp;
      smat[NX*j+i]     = qmat[NX*j+i]/tmp;
    } 

  for(i = 0; i<NX-1; ++i){
    tmp = fabs(pmat[(NX-1)*(NY-1)+i]);
    if(tmp<1) tmp = 1;
    rmat[(NX-1)*(NY-1)+i] = pmat[(NX-1)*(NY-1)+i]/tmp;
  } 

  for(j = 0; j<NY-1; ++j){
    tmp = fabs(qmat[NX*j+NX-1]);
    if(tmp<1) tmp = 1;
    smat[NX*j+NX-1] = qmat[NX*j+NX-1]/tmp;
  } 
}

double calc_Q_L1(int *M, int *N,
        double *yvec, double *Amatrix, double *xvec1, double *xvec2,
        double lambda_l1, double c, int *inc,
        double *yAz, double *AyAz, double *buffxvec1)
{
  int i;
  double term1, term2, term3, term4, term5, alpha = -1;

  /* x1 - x2 */
  dcopy_(N, xvec1, inc, buffxvec1, inc);
  daxpy_(N, &alpha, xvec2, inc, buffxvec1, inc);

  /* (y - A x2)'(y - A x2) */
  term1 = ddot_(M, yAz, inc, yAz, inc);
  /* (x1 - x2)'A'(y - A x2) */
  term2 = ddot_(N, AyAz, inc, buffxvec1, inc);
  /* (x1 - x2)'(x1 - x2) */
  term3 = ddot_(N, buffxvec1, inc, buffxvec1, inc);
  /* sum (x2_i) */
  term4 = dasum_(N, xvec2, inc);
  /* sum (x1-x2)_i */
  term5 = 0;
  for(i=0;i<(*N);++i) term5 += buffxvec1[i];

  return(term1/2-term2+c*term3/2+lambda_l1*term4 + lambda_l1*term5);
}

double calc_Q_L1_part(int *N,
		      double *xvec1, double *xvec2,
		      double lambda_l1, double c, int *inc,
		      double *AyAz, double *buffxvec1)
{
  int i;
  double term2, term3, term5, alpha = -1;

  /* x1 - x2 */
  dcopy_(N, xvec1, inc, buffxvec1, inc);
  daxpy_(N, &alpha, xvec2, inc, buffxvec1, inc);

  /* (x1 - x2)'A'(y - A x2) */
  term2 = ddot_(N, AyAz, inc, buffxvec1, inc);
  /* (x1 - x2)'(x1 - x2) */
  term3 = ddot_(N, buffxvec1, inc, buffxvec1, inc);
  /* sum (x1-x2)_i */
  term5 = 0;
  for(i=0;i<(*N);++i) term5 += buffxvec1[i];

  return(-term2+c*term3/2 + lambda_l1*term5);
}

double calc_Q(int *M, int *N, 
        double *yvec, double *Amatrix, double *xvec1, double *xvec2,
        double c, int *inc,
        double *yAz, double *AyAz, double *buffxvec1)
{
  double term1, term2, term3, alpha = -1;

  /* x1 - x2 */
  dcopy_(N, xvec1, inc, buffxvec1, inc);
  daxpy_(N, &alpha, xvec2, inc, buffxvec1, inc);

  /* (y - A x2)'(y - A x2) */
  term1 = ddot_(M, yAz, inc, yAz, inc);
  /* (x1 - x2)'A'(y - A x2) */
  term2 = ddot_(N, AyAz, inc, buffxvec1, inc);
  /* (x1 - x2)'(x1 - x2) */
  term3 = ddot_(N, buffxvec1, inc, buffxvec1, inc);

  return(term1/2-term2+c*term3/2);
}

double calc_F_L1(int *M, int *N,
        double *yvec, double *Amatrix,
        double *xvec, double lambda_l1,
        int *inc, double *buffvec)
{
  double term1, term2, alpha = -1, beta = 1;

  dcopy_(M, yvec, inc, buffvec, inc);

  dgemv_("N", M, N, &alpha, Amatrix, M, 
   xvec, inc, &beta, buffvec, inc);

  term1 = ddot_(M, buffvec, inc, buffvec, inc);
  term2 = dasum_(N, xvec, inc);

  return(term1/2+lambda_l1*term2);
}

void FGP(int *N, int NX, int NY,
	 double *bvec, double lambda_tv, int ITER, int *inc,
	 double *pmat, double *qmat, double *rmat, double *smat, 
	 double *npmat, double *nqmat, double *xvec)
{
  int i, iter, Ntmpp = (NX-1)*NY, Ntmpq = NX*(NY-1);
  double t = 1, tnew, alpha = 1/(8*lambda_tv),
    alpha1 = -lambda_tv, alpha2 = 1, beta;

  /* initialization */
  for(i = 0; i< Ntmpp; ++i) pmat[i]=0;
  for(i = 0; i< Ntmpq; ++i) qmat[i]=0;

  dcopy_(&Ntmpp, pmat, inc, rmat, inc);
  dcopy_(&Ntmpq, qmat, inc, smat, inc);

  /* iteration */

  for(iter = 0; iter < ITER; ++iter){

    tnew = (1+sqrt(1+4*t*t))/2;

    Lpq(NX, NY, rmat, smat, xvec);

    dscal_(N, &alpha1, xvec, inc);
    daxpy_(N, &alpha2, bvec, inc, xvec, inc);

    Lx(NX, NY, xvec, npmat, nqmat);

    /*[pnew,qnew] = P_pqiso((r+tmpr/(8*lambda)),(s+tmps/(8*lambda)),N);*/

    daxpy_(&Ntmpp, &alpha, npmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &alpha, nqmat, inc, smat, inc);

    P_pqiso(NX, NY, rmat, smat, npmat, nqmat);

    dcopy_(&Ntmpp, npmat, inc, rmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, smat, inc);

    beta = 1+((t-1)/tnew);
    dscal_(&Ntmpp, &beta, rmat, inc);
    dscal_(&Ntmpq, &beta, smat, inc);

    beta = -(t-1)/tnew;
    daxpy_(&Ntmpp, &beta, pmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &beta, qmat, inc, smat, inc);

    /* update */
    t = tnew;
    dcopy_(&Ntmpp, npmat, inc, pmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, qmat, inc);

  }

  /* P_positive(b-lambda_tv*L_pq(p,q)) */

  Lpq(NX, NY, pmat, qmat, xvec);

  dscal_(N, &alpha1, xvec, inc);
  daxpy_(N, &alpha2, bvec, inc, xvec, inc);

}

void FGP_L1(int *N, int NX, int NY,
	    double *bvec, double lambda_l1, double lambda_tv,
	    int ITER, int *inc,
	    double *pmat, double *qmat, double *rmat, double *smat, 
	    double *npmat, double *nqmat, double *xvec)
{
  int i, iter, Ntmpp = (NX-1)*NY, Ntmpq = NX*(NY-1);
  double t = 1, tnew,
    alpha1 = -lambda_tv, alpha2 = 1, alpha3 = 1/(8*lambda_tv), beta;

  /* initialization */
  for(i = 0; i< Ntmpp; ++i) pmat[i]=0;
  for(i = 0; i< Ntmpq; ++i) qmat[i]=0;

  dcopy_(&Ntmpp, pmat, inc, rmat, inc);
  dcopy_(&Ntmpq, qmat, inc, smat, inc);

  /* iteration */

  for(iter = 0; iter < ITER; ++iter){

    tnew = (1+sqrt(1+4*t*t))/2;

    /* P_positive(b-lambda_tv*L_pq(r,s)) */
   
    Lpq(NX, NY, rmat, smat, xvec);
    dscal_(N, &alpha1, xvec, inc);
    daxpy_(N, &alpha2, bvec, inc, xvec, inc);
    soft_threshold(xvec, *N, lambda_l1, xvec);
    
    /*  [tmpr,tmps] = Lx(tmp,N); */
    Lx(NX, NY, xvec, npmat, nqmat);

    /*[pnew,qnew] = P_pqiso((r+tmpr/(8*lambda)),(s+tmps/(8*lambda)),N);*/

    daxpy_(&Ntmpp, &alpha3, npmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &alpha3, nqmat, inc, smat, inc);

    P_pqiso(NX, NY, rmat, smat, npmat, nqmat);

    dcopy_(&Ntmpp, npmat, inc, rmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, smat, inc);

    beta = 1+(t-1)/tnew;
    dscal_(&Ntmpp, &beta, rmat, inc);
    dscal_(&Ntmpq, &beta, smat, inc);

    beta = -(t-1)/tnew;
    daxpy_(&Ntmpp, &beta, pmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &beta, qmat, inc, smat, inc);

    /* update */
    t = tnew;
    dcopy_(&Ntmpp, npmat, inc, pmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, qmat, inc);
  }

  /* P_positive(b-lambda_tv*L_pq(p,q)) */

  Lpq(NX, NY, pmat, qmat, xvec);
  dscal_(N, &alpha1, xvec, inc);
  daxpy_(N, &alpha2, bvec, inc, xvec, inc);
  soft_threshold(xvec, *N, lambda_l1, xvec);

}

void FGP_nonneg(int *N, int NX, int NY,
		double *bvec, double lambda_tv, int ITER, int *inc,
		double *pmat, double *qmat, double *rmat, double *smat, 
		double *npmat, double *nqmat, double *xvec)
{
  int i, iter, Ntmpp = (NX-1)*NY, Ntmpq = NX*(NY-1);
  double t = 1, tnew,
    alpha1 = -lambda_tv, alpha2 = 1, alpha3 = 1/(8*lambda_tv), beta;

  /* initialization */
  for(i = 0; i< Ntmpp; ++i) pmat[i]=0;
  for(i = 0; i< Ntmpq; ++i) qmat[i]=0;

  dcopy_(&Ntmpp, pmat, inc, rmat, inc);
  dcopy_(&Ntmpq, qmat, inc, smat, inc);

  /* iteration */

  for(iter = 0; iter < ITER; ++iter){

    tnew = (1+sqrt(1+4*t*t))/2;

    /* P_positive(b-lambda_tv*L_pq(r,s)) */
   
    Lpq(NX, NY, rmat, smat, xvec);
    dscal_(N, &alpha1, xvec, inc);
    daxpy_(N, &alpha2, bvec, inc, xvec, inc);
    P_positive(*N, xvec);

    /*  [tmpr,tmps] = Lx(tmp,N); */
    Lx(NX, NY, xvec, npmat, nqmat);

    /*[pnew,qnew] = P_pqiso((r+tmpr/(8*lambda)),(s+tmps/(8*lambda)),N);*/

    daxpy_(&Ntmpp, &alpha3, npmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &alpha3, nqmat, inc, smat, inc);

    P_pqiso(NX, NY, rmat, smat, npmat, nqmat);

    dcopy_(&Ntmpp, npmat, inc, rmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, smat, inc);

    beta = 1+(t-1)/tnew;
    dscal_(&Ntmpp, &beta, rmat, inc);
    dscal_(&Ntmpq, &beta, smat, inc);

    beta = -(t-1)/tnew;
    daxpy_(&Ntmpp, &beta, pmat, inc, rmat, inc);
    daxpy_(&Ntmpq, &beta, qmat, inc, smat, inc);

    /* update */
    t = tnew;
    dcopy_(&Ntmpp, npmat, inc, pmat, inc);
    dcopy_(&Ntmpq, nqmat, inc, qmat, inc);
  }

  /* P_positive(b-lambda_tv*L_pq(p,q)) */

  Lpq(NX, NY, pmat, qmat, xvec);
  dscal_(N, &alpha1, xvec, inc);
  daxpy_(N, &alpha2, bvec, inc, xvec, inc);
  P_positive(*N, xvec);

}

/* subroutines for mfista*/

void mfista_TV_core(double *yvec, double *Amat, 
		    int *M, int *N, int NX, int NY,
		    double lambda_tv, double cinit,
		    double *xvec,
		    struct RESULT *mfista_result)
{
  double *ytmp, *zvec, *xtmp, *xnew, *yAz, *AyAz, *yAx,
    *rmat, *smat, *npmat, *nqmat, 
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat,
    mu=1, munew, alpha = 1, gamma=0, tmpa, tvcost;
  int i, iter, inc = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  zvec  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  AyAz  = alloc_vector(*N);
  yAz   = alloc_vector(*M);
  ytmp  = alloc_vector(*M);
  xtmp  = alloc_vector(*N);

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  cost  = alloc_vector(MAXITER);

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  tvcost = TV(NX,NY,xvec);

  costtmp = calc_F_part(M, N, yvec, Amat, xvec, &inc, ytmp);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    calc_yAz(M, N,  yvec, Amat, zvec, &inc, yAz);
    dgemv_("T", M, N, &alpha, Amat, M, yAz, &inc, &gamma, AyAz, &inc);

    Qcore = calc_F_part(M, N, yvec, Amat, zvec, &inc, ytmp);
    
    for( i = 0; i < MAXITER; i++){

      dcopy_(N, AyAz, &inc, xtmp, &inc);
      tmpa = 1/c;
      dscal_(N, &tmpa, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

      FGP(N, NX, NY, xtmp, lambda_tv/c, FGPITER,
	  &inc, pmat, qmat, rmat, smat, npmat, nqmat, xnew);

      Fval = calc_F_part(M, N, yvec, Amat, xnew, &inc, ytmp);
      Qval = calc_Q_part(N, xnew, zvec, c, &inc, AyAz, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    tvcost = TV(NX, NY, xnew);

    Fval += lambda_tv*tvcost;

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
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

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

  mfista_result->M = (*M);
  mfista_result->N = (*N);
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = 0;
  mfista_result->lambda_sqtv = 0;
  mfista_result->lambda_tv = lambda_tv;

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

  for(i = 0;i< (*N);i++)
    if(fabs(xnew[i]) > 0)
      ++ mfista_result->N_active;

  mfista_result->sqtvcost = 0;
  mfista_result->tvcost = TV(NX, NY, xvec);
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda_tv*(mfista_result->tvcost);

  mfista_result->looe = 0;
  mfista_result->Hessian_positive = -1;
    
  free(yAx);

  /* clear memory */  

  free(AyAz);

  free(cost);

  free(pmat);
  free(qmat);
  free(npmat);
  free(nqmat);

  free(rmat);
  free(smat);

  free(xtmp);
  free(xnew);
  free(yAz);
  free(ytmp);
  free(zvec);

}

void mfista_L1_TV_core(double *yvec, double *Amat, 
		       int *M, int *N, int NX, int NY,
		       double lambda_l1, double lambda_tv, double cinit,
		       double *xvec,
		       struct RESULT *mfista_result)
{
  double *ytmp, *zvec, *xtmp, *xnew, *yAz, *AyAz, *yAx,
    *rmat, *smat, *npmat, *nqmat,
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat,
    mu=1, munew, alpha = 1, gamma=0, tmpa, l1cost, tvcost;
  int i, iter, inc = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  zvec  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  AyAz  = alloc_vector(*N);
  yAz   = alloc_vector(*M);
  ytmp  = alloc_vector(*M);
  xtmp  = alloc_vector(*N);

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  cost  = alloc_vector(MAXITER);

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  tvcost = TV(NX, NY, xvec);

  costtmp = calc_F_L1(M, N, yvec, Amat, xvec, lambda_l1, &inc, ytmp);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    calc_yAz(M, N,  yvec, Amat, zvec, &inc, yAz);
    dgemv_("T", M, N, &alpha, Amat, M, yAz, &inc, &gamma, AyAz, &inc);

    Qcore = calc_F_part(M, N, yvec, Amat, zvec, &inc, ytmp);

    for( i = 0; i < MAXITER; i++){

      dcopy_(N, AyAz, &inc, xtmp, &inc);
      tmpa = 1/c;
      dscal_(N, &tmpa, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

      FGP_L1(N, NX, NY, xtmp, lambda_l1/c, lambda_tv/c, FGPITER,
	     &inc, pmat, qmat, rmat, smat, npmat, nqmat, xnew);

      Fval = calc_F_part(M, N, yvec, Amat, xnew, &inc, ytmp);
      Qval = calc_Q_part(N, xnew, zvec, c, &inc, AyAz, xtmp);
      Qval += Qcore;

      /*    printf("c = %g, F = %g, G = %g\n",c,Fval,Qval);*/
      
      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    tvcost = TV(NX, NY, xnew);
    l1cost = dasum_(N, xnew, &inc);

    Fval += (lambda_l1*l1cost + lambda_tv*tvcost);

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
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

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
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_sqtv = 0;
  mfista_result->lambda_tv = lambda_tv;

  yAx  = alloc_vector(*M);
  calc_yAz(M, N, yvec, Amat, xvec, &inc, yAx);

  /* mean square error */
  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++)
    mfista_result->sq_error += yAx[i]*yAx[i];

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
  mfista_result->tvcost = TV(NX, NY, xvec);
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda_l1*(mfista_result->l1cost)
    + lambda_tv*(mfista_result->tvcost);

  mfista_result->looe = 0;
  mfista_result->Hessian_positive = -1;
    
  free(yAx);
  
  /* clear memory */

  free(AyAz);

  free(cost);

  free(npmat);
  free(nqmat);
  free(pmat);
  free(qmat);
  free(rmat);
  free(smat);

  free(xnew);
  free(xtmp);
  free(yAz);
  free(ytmp);
  free(zvec);
}

void mfista_L1_TV_core_nonneg(double *yvec, double *Amat, 
			      int *M, int *N, int NX, int NY,
			      double lambda_l1, double lambda_tv, double cinit,
			      double *xvec,
			      struct RESULT *mfista_result)
{
  double *ytmp, *zvec, *xtmp, *xnew, *yAz, *AyAz, *yAx,
    *rmat, *smat, *npmat, *nqmat,
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat, *ones,
    mu=1, munew, alpha = 1, gamma=0, tmpa, tvcost;
  int i, iter, inc = 1;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  zvec  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  AyAz  = alloc_vector(*N);
  yAz   = alloc_vector(*M);
  ytmp  = alloc_vector(*M);
  xtmp  = alloc_vector(*N);
  ones  = alloc_vector(*N);

  for(i = 0; i < (*N); ++i) ones[i]=1;

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  cost  = alloc_vector(MAXITER);

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  tvcost = TV(NX, NY, xvec);

  costtmp = calc_F_L1(M, N, yvec, Amat, xvec, lambda_l1, &inc, ytmp);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    calc_yAz(M, N,  yvec, Amat, zvec, &inc, yAz);
    dgemv_("T", M, N, &alpha, Amat, M, yAz, &inc, &gamma, AyAz, &inc);

    Qcore = calc_F_L1(M, N, yvec, Amat, zvec, lambda_l1, &inc, ytmp);

    for( i = 0; i < MAXITER; i++){

      dcopy_(N, ones, &inc, xtmp, &inc);
      tmpa = -lambda_l1/c;
      dscal_(N, &tmpa, xtmp, &inc);
      tmpa = 1/c;
      daxpy_(N, &tmpa, AyAz, &inc, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

      FGP_nonneg(N, NX, NY, xtmp, lambda_tv/c, FGPITER,
		 &inc, pmat, qmat, rmat, smat, npmat, nqmat, xnew);

      Fval = calc_F_L1(M, N, yvec, Amat, xnew, lambda_l1, &inc, ytmp);
      Qval = calc_Q_L1_part(N, xnew, zvec, lambda_l1, c, &inc, AyAz, xtmp);
      Qval += Qcore;
      
      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    tvcost = TV(NX, NY, xnew);

    Fval += lambda_tv*tvcost;

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
     
    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

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
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_sqtv = 0;
  mfista_result->lambda_tv = lambda_tv;

  yAx  = alloc_vector(*M);
  calc_yAz(M, N, yvec, Amat, xvec, &inc, yAx);

  /* mean square error */
  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++)
    mfista_result->sq_error += yAx[i]*yAx[i];

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
  mfista_result->tvcost = TV(NX, NY, xvec);
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda_l1*(mfista_result->l1cost)
    + lambda_tv*(mfista_result->tvcost);

  mfista_result->looe = 0;
  mfista_result->Hessian_positive = -1;
    
  free(yAx);

  /* clear memory */

  free(ytmp);

  free(xtmp);

  free(zvec);
  free(xnew);

  free(AyAz);
  free(yAz);

  free(ones);
  free(pmat);
  free(qmat);
  free(npmat);
  free(nqmat);

  free(rmat);
  free(smat);
  free(cost);

}

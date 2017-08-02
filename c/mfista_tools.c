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

/* memory allocation of matrix and vectors */

int *alloc_int_vector(int length)
{
    return malloc(sizeof(int)*length);
}

double *alloc_vector(int length)
{
    return malloc(sizeof(double)*length);
}

double *alloc_matrix(int height, int width)
{
    return malloc(sizeof(double) * height * width);
}

void clear_matrix(double *matrix, int height, int width)
{
    memset(matrix, 0, sizeof(double) * height * width);
}

/* file i_o*/


FILE* fopenr(char* fn){
  FILE *fp;

  fp = fopen(fn,"r");
  if (fp==NULL){
    fprintf(stderr," --- Can't fopen(r) %s\n",fn);
    exit(1);
  }
  return(fp);
}

FILE* fopenw(char* fn){
  FILE *fp;

  fp = fopen(fn,"w");
  if (fp==NULL){
    fprintf(stderr," --- Can't fopen(r) %s\n",fn);
    exit(1);
  }
  return(fp);
}

int read_int_vector(char *fname, int length, int *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(int), length, fp);
  fclose(fp);

  return(n);
}

int read_V_vector(char *fname, int length, double *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(double), length, fp);
  fclose(fp);

  return(n);
}

unsigned long read_A_matrix(char *fname, int height, int width, double *matrix)
{
  FILE *fp;
  int i;
  double tmplength;
  unsigned long n, tmp;

  tmplength = (double)ULONG_MAX;
  tmplength /= (double)height;
  tmplength /= (double)width;

  if(tmplength > 1){

    n = height;
    n *= width;
    printf("Reading %d x %d = %ld entries of matrix A.\n",height, width, n);

    fp = fopenr(fname);  
    tmp = fread(matrix, sizeof(double), n, fp);
    fclose(fp);
  }
  else{
    n = 0;
    fp = fopenr(fname);  
    for(i=0;i < height; i++){
      tmp = fread(matrix + n, sizeof(double), width, fp);
      n += tmp;
      if(i%1000==0)
  printf("reading %d line of A.\n",i);
    }
    fclose(fp);
  }
      
  return(n);
}

int write_X_vector(char *fname, int length, double *vector)
{
  FILE *fp;
  int n;

  fp = fopenw(fname);
  n  = fwrite(vector, sizeof(double), length, fp);
  fclose(fp);

  return(n);
}

/* matrix operation*/

void transpose_matrix(double *matrix, int origheight, int origwidth)
/* put transpose of matrix to original matrix space */
{
  int i, j;
  double *tmpmat;

  tmpmat = alloc_matrix(origwidth, origheight);

  for(i=0; i< origheight;i++){
    for(j=0; j< origwidth;j++)
      tmpmat[j*origheight+i] = matrix[i*origwidth+j];
  }

  for(i=0; i< origwidth*origheight;i++)
    matrix[i] = tmpmat[i];

  free(tmpmat);
}

/* subroutines for mfista*/

void calc_yAz(int *M, int *N,
	      double *yvec, double *Amat, double *zvec, double *yAz)
{
  int inc = 1;
  double alpha = -1, beta = 1;

  /* y - A x2 */
  dcopy_(M, yvec, &inc, yAz, &inc);
  dgemv_("N", M, N, &alpha, Amat, M, zvec, &inc, &beta, yAz, &inc);
}

double calc_F_part(int *M, int *N,
		   double *yvec, double *Amatrix,
		   double *xvec, double *buffvec)
{
  int inc = 1;
  double term1, alpha = -1, beta = 1;

  dcopy_(M, yvec, &inc, buffvec, &inc);

  dgemv_("N", M, N, &alpha, Amatrix, M, 
	 xvec, &inc, &beta, buffvec, &inc);

  term1 = ddot_(M, buffvec, &inc, buffvec, &inc);

  return(term1/2);
}

double calc_Q_part(int *N, double *xvec1, double *xvec2,
		   double c, double *AyAz, double *buffxvec1)
{
  int inc = 1;
  double term2, term3, alpha = -1;

  /* x1 - x2 */
  dcopy_(N, xvec1, &inc, buffxvec1, &inc);
  daxpy_(N, &alpha, xvec2, &inc, buffxvec1, &inc);

  /* (x1 - x2)'A'(y - A x2) */
  term2 = ddot_(N, AyAz, &inc, buffxvec1, &inc);
  /* (x1 - x2)'(x1 - x2) */
  term3 = ddot_(N, buffxvec1, &inc, buffxvec1, &inc);

  return(-term2+c*term3/2);
}

/* soft thresholding */

void soft_threshold_nonneg(double *vector, int length, double eta, 
			   double *newvec)
{
    int i;
    for (i = 0; i < length; i++){
      if(vector[i] < eta)
	newvec[i] = 0;
      else
	newvec[i] = vector[i] - eta;
    }
}

void soft_threshold(double *vector, int length, double eta, 
		    double *newvec)
{
    int i;
    for (i = 0; i < length; i++){
      if(vector[i] >= eta)
	newvec[i] = vector[i] - eta;
      else if(vector[i] < eta && vector[i] > -eta)
	newvec[i] = 0;
      else
	newvec[i] = vector[i] + eta;
    }
}

int find_active_set(int N, double *xvec, int *indx_list)
{
  int i, N_active;

  N_active = 0;
  
  for(i = 0;i < N;i++){
    if(fabs(xvec[i]) > 0){
	indx_list[N_active] = i;
	N_active++;
      }
  }

  return(N_active);
}

/* index transform */

int i2r(int i, int NX)
{
  return(i%NX);
}

int i2c(int i, int NX)
{
  double tmp1, tmp2;

  tmp1 = (double)i;
  tmp2 = (double)NX;
  
  return((int)(ceil(((tmp1+1)/tmp2))-1));
}

int rc2i(int r, int c, int NX)
{
  return( r + NX*c);
}

/* Some routines for computing LOOE */

double *shrink_A(int M, int N, int N_active, int *indx_list,
	      double *Amat)
{
  int i, j, k;
  double *Amat_s;

  Amat_s = alloc_matrix(M, N_active);

  for(j = 0; j < N_active; j++){
    k = indx_list[j];
    for(i = 0; i < M; i++) Amat_s[j*M+i] = Amat[k*M+i];
  }

  return(Amat_s);
}

int solve_lin_looe(int *NA, int *NB, double *Hessian, double *B)
/* Solve A*X = B
   A is a symmetric real matrix with *NA x *NA.
   B is a real matrix with *NA x *NB.
   UpLo is "U" or "L." It tells which part of A is used.
   The function is void and the result is stored in B. */
{
  int  info, lda, ldb;
  char UpLo[2] = {'L','\0'};;
    
    lda  = *NA;
    ldb  = lda;
    
    info = 0;

    printf("Solving a linear equation.\n");

    dposv_(UpLo, NA, NB, Hessian, &lda, B, &ldb, &info );

    printf("solved.\n");

    if (info < 0)      printf("DPOSV: The matrix had an illegal value.\n");
    else if (info > 0) printf("DPOSV: The Hessian matrix is not positive definite.\n");

    return(info);
}

double compute_LOOE_core(int *M, int N_active, 
			 double *yvec, double *Amat, double *xvec,
			 double *yAx,  double *Amat_s, double *Hessian)
{
  int i, j, m, n_s;
  double LOOE, *At, *dvec, tmp, info;

  m   = *M;          /* size of Hessian */
  n_s = N_active;  /* number of columns of Amat_s */

  At  = alloc_matrix(n_s,m);

  dvec = alloc_vector(m);

  for(i = 0;i < m; i++)
    for(j = 0;j < n_s; j++)
      At[n_s*i + j] = Amat_s[m*j+i];

  info = solve_lin_looe(&n_s, &m, Hessian, At);

  if(info == 0){

    for(i = 0;i < m;i++){
      dvec[i]=0;
      for(j = 0;j < n_s;j++)
	dvec[i]+= Amat_s[m*j+i]*At[n_s*i+j];
    }
    
    LOOE = 0;
    
    for(i=0;i<m;++i){
      tmp = yAx[i]/(1-dvec[i]);
      LOOE += tmp*tmp;
    }
    LOOE /=(2*((double)m));

    free(dvec);
    free(At);

    return(LOOE);
  }
  else{
    return(-1.0);
  }
}

void show_result(FILE *fid, char *fname, struct RESULT *mfista_result)
{
  fprintf(fid,"\n\n");

  fprintf(fid,"Output of %s.\n",fname);

  fprintf(fid,"\n\n");
  
  fprintf(fid," Size of the problem:\n\n");
  fprintf(fid," size of input vector:   %d\n",mfista_result->M);
  fprintf(fid," size of output vector:  %d\n",mfista_result->N);
  if(mfista_result->NX!=0)
    fprintf(fid," size of image:          %d x %d\n",
	    mfista_result->NX,mfista_result->NY);
  
  fprintf(fid,"\n\n");
  fprintf(fid," Problem Setting:\n\n");

  if(mfista_result->nonneg == 1)
    fprintf(fid," x is a nonnegative vector.\n\n");
  else if (mfista_result->nonneg == 0)
    fprintf(fid," x is a real vector (takes 0, positive, and negative value).\n\n");

  fprintf(fid," input vector file:      %s\n", mfista_result->v_fname);
  fprintf(fid," input matrix file:      %s\n", mfista_result->A_fname);


  if(mfista_result->in_fname != NULL)
    fprintf(fid," x was initialized with: %s\n", mfista_result->in_fname);

  if(mfista_result->lambda_l1 != 0)
    fprintf(fid," Lambda_1:               %e\n", mfista_result->lambda_l1);

  if(mfista_result->lambda_tsv != 0)
    fprintf(fid," Lambda_TSV:             %e\n", mfista_result->lambda_tsv);

  if(mfista_result->lambda_tv != 0)
    fprintf(fid," Lambda_TV:              %e\n", mfista_result->lambda_tv);

  fprintf(fid," MAXITER:                %d\n", mfista_result->maxiter);

  fprintf(fid,"\n\n");
  fprintf(fid," Results:\n\n");

  fprintf(fid," # of iterations:        %d\n", mfista_result->ITER);
  fprintf(fid," cost:                   %e\n", mfista_result->finalcost);
  fprintf(fid," computaion time[sec]:   %e\n\n", mfista_result->comp_time);
  
  fprintf(fid," x is saved to:          %s\n", mfista_result->out_fname);
  fprintf(fid,"\n");

  fprintf(fid," # of nonzero pixels:    %d\n", mfista_result->N_active);
  fprintf(fid," Squared Error (SE):     %e\n", mfista_result->sq_error);
  fprintf(fid," Mean SE:                %e\n", mfista_result->mean_sq_error);

  if(mfista_result->lambda_l1 != 0)
    fprintf(fid," L1 cost:                %e\n", mfista_result->l1cost);

  if(mfista_result->lambda_tsv != 0)
    fprintf(fid," TSV cost:               %e\n", mfista_result->tsvcost);

  if(mfista_result->lambda_tv != 0)
    fprintf(fid," TV cost:                %e\n", mfista_result->tvcost);

  fprintf(fid,"\n");
  
  if(mfista_result->Hessian_positive ==1)
    fprintf(fid," LOOE:                   %e\n", mfista_result->looe);
  else if (mfista_result->Hessian_positive ==0)
    fprintf(fid," LOOE:    Could not be computed because Hessian was not positive definite.\n");
  else if (mfista_result->Hessian_positive == -1)
    fprintf(fid," LOOE:    Did not compute LOOE.\n");

  fprintf(fid,"\n");

}

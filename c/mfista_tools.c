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

#ifdef __APPLE__
#include <sys/time.h>
#endif

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


/* TV */

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

/* TSV */

double TSV(int NX, int NY, double *xvec)
{
  int i, j;
  double tsv = 0;

  for(i = 0; i < NX-1; ++i)
    for(j = 0; j < NY-1; ++j){
      tsv += pow((xvec[NX*j+i]-xvec[NX*j+i+1]),2.0);
      tsv += pow((xvec[NX*j+i]-xvec[NX*(j+1)+i]),2.0);
    }

  for(i = 0; i < NX-1; ++i)
    tsv += pow((xvec[NX*(NY-1)+i]-xvec[NX*(NY-1)+i+1]),2.0);

  for(j = 0; j < NY-1; ++j)
    tsv += pow((xvec[NX*j+(NX-1)]-xvec[NX*(j+1)+(NX-1)]),2.0);

  return(tsv);
}

void d_TSV(int NX, int NY, double *xvec, double *dvec)
{
  int i, j;

  for(j = 0; j < NY; j++) dvec[NX*j+NX-1] = 0;

  for(i = 0; i < NX-1; i++)
    for(j = 0; j < NY; j++)
      dvec[NX*j+i] = 2*(xvec[NX*j+i]-xvec[NX*j+i+1]);

  for(i = 1; i < NX; i++)
    for(j = 0; j < NY; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*j+i-1]);

  for(i = 0; i < NX; i++)
    for(j = 0; j < NY-1; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*(j+1)+i]);

  for(i = 0; i < NX; i++)
    for(j = 1; j < NY; j++)
      dvec[NX*j+i] += 2*(xvec[NX*j+i]-xvec[NX*(j-1)+i]);
}

/* results */

void calc_result(double *yvec, double *Amat,
		 int *M, int *N, int NX, int NY,
		 double lambda_l1, double lambda_tv, double lambda_tsv,
		 double *xvec, int nonneg_flag, int looe_flag,
		 struct RESULT *mfista_result)
{
  int i;
  double *yAx, tmpa;

  printf("summarizing result.\n");

  /* allocate memory space start */ 

  yAx  = alloc_vector(*M);
  mfista_result->residual = alloc_vector(*M);
  
  /* summary of results */

  mfista_result->M = (*M);
  mfista_result->N = (*N);
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_tv = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;

  calc_yAz(M, N, yvec, Amat, xvec, yAx);

  /* mean square error */
  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++){
    mfista_result->sq_error += yAx[i]*yAx[i];
    mfista_result->residual[i] = yAx[i];
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

  mfista_result->finalcost = (mfista_result->sq_error)/2;

  if(lambda_l1 > 0)
    mfista_result->finalcost += lambda_l1*(mfista_result->l1cost);

  if(lambda_tsv > 0){
    mfista_result->tsvcost = TSV(NX, NY, xvec);
    mfista_result->finalcost += lambda_tsv*(mfista_result->tsvcost);
  }
  else if (lambda_tv > 0){
    mfista_result->tvcost = TV(NX, NY, xvec);
    mfista_result->finalcost += lambda_tv*(mfista_result->tvcost);
  }

  /* computing LOOE */

  if(looe_flag == 1 && lambda_tv ==0 ){
    if(lambda_tsv == 0){
      mfista_result->looe_m = compute_LOOE_L1(M, N, lambda_l1, yvec, Amat, xvec, yAx,
					    &(mfista_result->looe_m), &(mfista_result->looe_std));
      printf("%le\n",mfista_result->looe_m);
    }
    else
      mfista_result->looe_m = compute_LOOE_L1_TSV(M, N, NX, NY, lambda_l1, lambda_tsv,
						yvec, Amat, xvec, yAx,
						&(mfista_result->looe_m), &(mfista_result->looe_std));
    if(mfista_result->looe_m == -1){
      mfista_result->Hessian_positive = 0;
      mfista_result->looe_m = 0;
    }
    else{
      mfista_result->Hessian_positive = 1;
    }
  }
  else{
    mfista_result->looe_m = 0;
    mfista_result->Hessian_positive = -1;
  }

  /* clear memory */
  
  free(yAx);
}

void show_io_fnames(FILE *fid, char *fname, struct IO_FNAMES *mfista_io)
{
  fprintf(fid,"\n\n");

  fprintf(fid,"IO files of %s.\n",fname);

  fprintf(fid,"\n\n");
  
  if ( mfista_io->fft == 0){
    fprintf(fid," input vector file:      %s\n", mfista_io->v_fname);
    fprintf(fid," input matrix file:      %s\n", mfista_io->A_fname);
  }
  else 
    fprintf(fid," FFTW file:              %s\n", mfista_io->fft_fname);  

  if(mfista_io->in_fname != NULL)
    fprintf(fid," x was initialized with: %s\n", mfista_io->in_fname);

  if(mfista_io->out_fname != NULL)
    fprintf(fid," x is saved to:          %s\n", mfista_io->out_fname);

  fprintf(fid,"\n");
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
  
  if(mfista_result->Hessian_positive ==1){
    fprintf(fid," LOOE:(mean)             %e\n", mfista_result->looe_m);
    fprintf(fid," LOOE:(std)              %e\n", mfista_result->looe_std);
  }
  else if (mfista_result->Hessian_positive ==0)
    fprintf(fid," LOOE:    Could not be computed because Hessian was not positive definite.\n");
  else if (mfista_result->Hessian_positive == -1)
    fprintf(fid," LOOE:    Did not compute LOOE.\n");

  fprintf(fid,"\n");

}

/* utility for time measurement */
void get_current_time(struct timespec *t) {
#ifdef __APPLE__
  struct timeval tv;
  struct timezone tz;
  int status = gettimeofday(&tv, &tz);
  if (status == 0) {
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

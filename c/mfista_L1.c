/* 
   Copyright (C) 2015   Shiro Ikeda <shiro@ism.ac.jp>

   This is file 'mfista.c'. An optimization algorithm for imaging of
   interferometry. The idea of the algorithm was from the following
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

void usage(char *s)
{
  printf("%s <int m> <intl n> <V fname> <A fname> <double lambda_l1> <double c> <X outfile> {X initfile} {-t} {-looe} {-nonneg} {-log log_fname}\n\n",s);
  printf("  <int m>: number of row of A.\n");
  printf("  <int n>: number of column of A.\n");
  printf("  <V fname>: file name of V.\n");
  printf("  <A fname>: file name of A.\n");
  printf("  <double lambda_l1>: value of lambda_l1. Positive.\n");
  printf("  <double c>: value of c. Positive.\n");
  printf("  <X outfile>: file name to write X.\n\n");

  printf(" Options.\n\n");

  printf("  {X initfile}: file name of X for initialization.(optional).\n");
  printf("  {-t}: use this option if A is stored with row major mode.\n");
  printf("  {-looe}: Compute approximation of LOOE.\n");
  printf("  {-nonneg}: Use this option if x is nonnegative.\n");
  printf("  {-log log_fname}: Specify log file.\n\n");

  printf(" This program solves \n\n");

  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1\n\n");

  printf(" and write x to <X out file>\n\n");

  printf(" If {-nonneg} option is used, x vector is restricted to be nonnegative.\n\n");

  printf(" c is a parameter used for stepsize. Large c makes the algorithm\n");
  printf(" stable but slow. Around 500000 is fine.\n\n");

  printf(" Files are binary. Read and Write with fread() and fwrite().\n");
  printf(" A is col major. This is C program but blas is based on fortran.\n\n");
  exit(1);
}

int main(int argc, char *argv[])
{
  double *y, *A, *xvec, cinit = CINIT, lambda_l1, s_t, e_t;
  char init_fname[1024],fname[1024],log_fname[1024];
  int i, M, N, iter, trans_flag = 0, init_flag = 0, nonneg_flag = 0,
    looe_flag = 0, log_flag = 0;
  unsigned long tmpdnum, dnum;
  struct IO_FNAMES mfista_io;
  struct RESULT    mfista_result;
  struct timespec time_spec1, time_spec2;
  FILE* log_fid;

  /* options */

  if (argc<8) usage(argv[0]);

  for(i=8; i<argc ; i++){
    if(strcmp(argv[i],"-t") == 0){
      trans_flag = 1; 
    }
    else if(strcmp(argv[i],"-looe") == 0){
      looe_flag = 1; 
    }
    else if(strcmp(argv[i],"-log") == 0){
      log_flag = 1;

      ++i;
      strcpy(log_fname,argv[i]);
    }
    else if(strcmp(argv[i],"-nonneg") == 0){
      nonneg_flag = 1;
    }
    else{
      init_flag = 1;
      strcpy(init_fname,argv[i]);
    }
  }

  M = atoi(argv[1]);
  printf("M is %d\n",M);

  N = atoi(argv[2]);
  printf("N is %d\n",N);

  /* allocate memory space start */ 

  y = alloc_vector(M);
  A = alloc_matrix(M,N);

  xvec = alloc_vector(N);

  /* initialize xvec */

  if (init_flag ==1){ 

    printf("Initializing x with %s.\n",init_fname);
    dnum = read_V_vector(init_fname, N, xvec);

    if(dnum != N)
      printf("Number of read data is shorter than expected.\n");
  }
  else
    clear_matrix(xvec, N, 1);

  /* read data matrix start */

  sprintf(fname,"%s",argv[3]);
  dnum = read_V_vector(fname, M, y);

  if(dnum != M) 
    printf("Number of read data is shorter than expected in %s.\n",
	   argv[3]);

  sprintf(fname,"%s",argv[4]);
  dnum = read_A_matrix(fname, M, N, A);

  tmpdnum = M;
  tmpdnum *= N;

  if(dnum != tmpdnum){
    printf("Number of read data is shorter than expected in %s.\n",
	   argv[4]);

    printf("expected num is %ld but it only had %ld.\n",
	   tmpdnum,dnum);
  }

  lambda_l1 = atof(argv[5]);
  printf("lambda_l1 = %g\n",lambda_l1);
 
  cinit = atof(argv[6]);
  printf("c = %g\n",cinit);

  if (nonneg_flag == 1)
    printf("x is nonnegative.\n");

  if (log_flag ==1)
    printf("Log will be saved to %s.\n",log_fname);

  if (looe_flag ==1)
    printf("Approximation of LOOE will be computed.\n\n");
  else
    printf("\n");

  if (trans_flag ==1){ 
    transpose_matrix(A, N, M);
  }

  /* preparation end */

  get_current_time(&time_spec1);

  /* main loop */

  iter = mfista_L1_TSV_core(y, A, &M, &N, 0, 0, lambda_l1, 0, cinit, xvec, nonneg_flag);

  get_current_time(&time_spec2);

  write_X_vector(argv[7], N, xvec);

  /* main loop end */

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;
  
  mfista_result.comp_time = e_t-s_t;
  mfista_result.ITER = iter;
  mfista_result.nonneg = nonneg_flag;

  calc_result(y, A, &M, &N, 0, 0, lambda_l1, 0, 0,
	      xvec, nonneg_flag, looe_flag, &mfista_result);
  
  /* output log */

  mfista_io.fft      = 0;
  mfista_io.fft_fname = NULL;
  mfista_io.v_fname = argv[3];
  mfista_io.A_fname = argv[4];

  if(init_flag == 1)
    mfista_io.in_fname = init_fname;
  else
    mfista_io.in_fname = NULL;

  mfista_io.out_fname = argv[7];
  show_io_fnames(stdout,argv[0],&mfista_io);
  show_result(stdout,argv[0],&mfista_result);

  if(log_flag == 1){
    log_fid = fopenw(log_fname);
    show_io_fnames(log_fid,argv[0],&mfista_io);
    show_result(log_fid,argv[0],&mfista_result);
    fclose(log_fid);
  }
  
  /* clear memory */

  free(y);
  free(A);
  free(xvec);

  return(0);
}

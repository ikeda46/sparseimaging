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
  printf("%s <fftw_data fname> <double lambda_l1> <double lambda_tsv> <double c> <X outfile> {X initfile} {-nonneg} {-log log_fname}\n\n",s);
  printf("  <fftw_data fname>: file name of fftw_file.\n");
  printf("  <double lambda_l1>: value of lambda_l1. Positive.\n");
  printf("  <double lambda_tsv>: value of lambda_tsv. Positive.\n");
  printf("  <double c>: value of c. Positive.\n");
  printf("  <X outfile>: file name to write X.\n\n");

  printf(" Options.\n\n");
    
  printf("  {X initfile}: file name of X for initialization.(optional).\n");
  printf("  {-nonneg}: Use this option if x is nonnegative.\n");
  printf("  {-log log_fname}: Specify log file.\n\n");

  printf(" This program solves the following problem with FFT\n\n");

  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tsv TSV(x)\n\n");

  printf(" If {-nonneg} option is used, x vector is restricted to be nonnegative.\n\n");

  printf(" c is a parameter used for stepsize. Larger the algorithm is stable. \n");
  printf(" but too large c makes convergence slow. Around 50000 is fine.\n\n");

  printf(" and writeh x to <X out file>\n\n");

  printf(" Files are binary. Read and Write with fread() and fwrite().\n");
  exit(1);
}

int main(int argc, char *argv[]){
 
  int M, NN, NX, NY, NY_h, dnum, i, j, *u_idx, *v_idx,
    init_flag = 0, log_flag = 0, nonneg_flag = 0;

  char init_fname[1024],fftw_fname[1024],log_fname[1024];
  double *y_r, *y_i, *noise_stdev, *xvec, 
    *mask_h, *mask, cinit = CINIT, lambda_l1, lambda_tsv, s_t, e_t;
  
  doublecomplex *yf; 
  struct RESULT mfista_result;
  struct timespec time_spec1, time_spec2;
  FILE *fftw_fp, *log_fid;

  /* options */

  if (argc<6) usage(argv[0]);

  for(i=6; i<argc ; i++){
    if(strcmp(argv[i],"-log") == 0){
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

  /* read fftw_data */

  strcpy(fftw_fname,argv[1]);
  
  fftw_fp = fopenr(fftw_fname);

  if (fscanf(fftw_fp, "M  = %d\n", &M)  !=1){exit(0);}
  if (fscanf(fftw_fp, "NX = %d\n", &NX) !=1){exit(0);}
  if (fscanf(fftw_fp, "NY = %d\n", &NY) !=1){exit(0);}
  if (fscanf(fftw_fp, "\n") !=0)            {exit(0);}
  if (fscanf(fftw_fp, "u, v, y_r, y_i, noise_std_dev\n") !=0) {exit(0);}
  if (fscanf(fftw_fp, "\n") !=0)            {exit(0);}

  printf("number of u-v points:  %d\n",M);
  printf("X-dim of image:        %d\n",NX);
  printf("Y-dim of image:        %d\n",NY);
  u_idx = alloc_int_vector(M);
  v_idx = alloc_int_vector(M);

  y_r    = alloc_vector(M);
  y_i    = alloc_vector(M);
  noise_stdev = alloc_vector(M);
  
  for(i = 0;i<M;++i){
    if(fscanf(fftw_fp, "%d, %d, %lf, %lf, %lf\n",
	      u_idx+i,v_idx+i,y_r+i,y_i+i,noise_stdev+i)!=5){
      printf("cannot read data.\n");
      exit(0);
    }
  }

  fclose(fftw_fp);


  /* initialize xvec */

  NN = NX*NY;

  xvec = alloc_vector(NN);

  if (init_flag ==1){ 

    printf("Initializing x with %s.\n",init_fname);
    dnum = read_V_vector(init_fname, NN, xvec);

    if(dnum != NN)
      printf("Number of read data is shorter than expected.\n");
  }
  else{
    clear_matrix(xvec, NN, 1);
  }

  /* read parameters */

  lambda_l1 = atof(argv[2]);
  printf("lambda_l1 = %g\n",lambda_l1);

  lambda_tsv = atof(argv[3]);
  printf("lambda_tsv = %g\n",lambda_tsv);

  cinit = atof(argv[4]);
  printf("c = %g\n",cinit);

  if (nonneg_flag == 1)
    printf("x is nonnegative.\n");

  if (log_flag ==1)
    printf("Log will be saved to %s.\n",log_fname);

  printf("\n");

  /* set parameters */

  NY_h = (int)floor(NY/2)+1;

  mask_h = alloc_vector(NX*NY_h);
  mask   = alloc_vector(NX*NY);

  yf     = (doublecomplex*) malloc(NX*NY*sizeof(doublecomplex));

  for(i=0;i<NX;++i) for(j=0;j<NY;++j){
      yf[NY*i+j].r = 0.0;
      yf[NY*i+j].i = 0.0;
    }

  for(i=0;i<M;++i){
    yf[NY*(u_idx[i]) + (v_idx[i])].r = y_r[i]/noise_stdev[i];
    yf[NY*(u_idx[i]) + (v_idx[i])].i = y_i[i]/noise_stdev[i];
    mask[NY*(u_idx[i]) + (v_idx[i])]  = 1/noise_stdev[i];
  }

  for(i=0;i<NX;++i) for(j=0;j<floor(NY/2)+1;++j){
      mask_h[NY_h*i+j] = mask[NY*i+j];
    }

  free(u_idx);
  free(v_idx);
  free(y_r);
  free(y_i);
  free(mask);

  printf("hi\n");

  /* preparation end */

  clock_gettime(CLOCK_MONOTONIC, &time_spec1);

  /* main loop */

  mfista_L1_TSV_core_fftw(yf, mask_h, M, &NN, NX, NY,
			  lambda_l1, lambda_tsv, cinit, xvec, nonneg_flag, 
			  &mfista_result);

  clock_gettime(CLOCK_MONOTONIC, &time_spec2);

  write_X_vector(argv[5], NN, xvec);

  s_t = (double)time_spec1.tv_sec + (10e-9)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-9)*(double)time_spec2.tv_nsec;

  mfista_result.comp_time = e_t-s_t;
  
  /* main end */

  /* output log */

  mfista_result.nonneg = nonneg_flag;
  mfista_result.v_fname = argv[1];
  mfista_result.A_fname = NULL;

  if(init_flag == 1)
    mfista_result.in_fname = init_fname;
  else
    mfista_result.in_fname = NULL;

  mfista_result.out_fname = argv[5];  
  show_result(stdout,argv[0],&mfista_result);

  if(log_flag == 1){
    log_fid = fopenw(log_fname);
    show_result(log_fid,argv[0],&mfista_result);
    fclose(log_fid);
  }

  /* clear memory */

  free(mask_h);
  free(noise_stdev);
  free(yf);
  free(xvec);
 
  return 0;
}

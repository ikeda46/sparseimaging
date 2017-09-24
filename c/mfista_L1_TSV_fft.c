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
  printf("%s <fft_data fname> <double lambda_l1> <double lambda_tsv> <double c> <X outfile> {X initfile} {-nonneg} {-fftw_measure} {-log log_fname}\n\n",s);
  printf("  <fft_data fname>: file name of fft_file.\n");
  printf("  <double lambda_l1>: value of lambda_l1. Positive.\n");
  printf("  <double lambda_tsv>: value of lambda_tsv. Positive.\n");
  printf("  <double c>: value of c. Positive.\n");
  printf("  <X outfile>: file name to write X.\n\n");

  printf(" Options.\n\n");
    
  printf("  {X initfile}: file name of X for initialization.(optional).\n");
  printf("  {-nonneg}: Use this option if x is nonnegative.\n");
  printf("  {-fftw_measure}: Let fftw make plan with FFTW_MEASURE.\n");
  printf("  {-log log_fname}: Specify log file.\n\n");

  printf(" This program solves the following problem with FFT\n\n");

  printf(" argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tsv TSV(x)\n\n");

  printf(" If {-nonneg} option is used, x vector is restricted to be nonnegative.\n\n");

  printf(" c is a parameter used for stepsize. Larger the algorithm is stable. \n");
  printf(" but too large c makes convergence slow. Around 50000 is fine.\n\n");

  printf(" and write x to <X out file>\n\n");

  exit(1);
}

int main(int argc, char *argv[]){

  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  int M, NN, NX, NY, dnum, i, iter, *u_idx, *v_idx,
    init_flag = 0, log_flag = 0, nonneg_flag = 0;

  char init_fname[1024],fftw_fname[1024],log_fname[1024];
  double *y_r, *y_i, *noise_stdev, *xinit, *xout, *mask,
    cinit = CINIT, lambda_l1, lambda_tsv, s_t, e_t;

  fftw_complex *yf;

  struct IO_FNAMES mfista_io;
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
    else if(strcmp(argv[i],"-fftw_measure") == 0){
      fftw_plan_flag = FFTW_MEASURE;
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

  /* initialize xout */

  NN = NX*NY;

  /* allocate memory */

  xinit = alloc_vector(NN);
  xout  = alloc_vector(NN);
  yf   = (fftw_complex*) fftw_malloc(NN*sizeof(fftw_complex));
  mask = alloc_vector(NN);

  if (init_flag ==1){ 

    printf("Initializing x with %s.\n",init_fname);
    dnum = read_V_vector(init_fname, NN, xinit);

    if(dnum != NN)
      printf("Number of read data is shorter than expected.\n");
  }
  else{
    clear_matrix(xinit, NN, 1);
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

  idx2mat(M, NX, NY, u_idx, v_idx, y_r, y_i, noise_stdev, yf, mask);

  /* preparation end */

  get_current_time(&time_spec1);

  /* main loop */

  iter = mfista_L1_TSV_core_fft(NX, NY, yf, mask, lambda_l1, lambda_tsv, cinit, xinit, xout,
				nonneg_flag, fftw_plan_flag);

  get_current_time(&time_spec2);

  write_X_vector(argv[5], NN, xout);

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result.ITER = iter;
  mfista_result.comp_time = e_t-s_t;
  mfista_result.nonneg = nonneg_flag;

  calc_result_fft(M, NX, NY, yf, mask, lambda_l1, 0, lambda_tsv, xout, &mfista_result);

  /* main end */

  /* output log */

  mfista_io.fft       = 1;
  mfista_io.fft_fname = argv[1];
  mfista_io.v_fname = NULL;
  mfista_io.A_fname = NULL;

  if(init_flag == 1)
    mfista_io.in_fname = init_fname;
  else
    mfista_io.in_fname = NULL;

  mfista_io.out_fname = argv[5];
  show_io_fnames(stdout, argv[0], &mfista_io);
  show_result(stdout, argv[0], &mfista_result);

  if(log_flag == 1){
    log_fid = fopenw(log_fname);
    show_io_fnames(log_fid, argv[0], &mfista_io);
    show_result(log_fid, argv[0], &mfista_result);
    fclose(log_fid);
  }

  /* clear memory */

  free(u_idx);
  free(v_idx);
  free(y_r);
  free(y_i);
  free(noise_stdev);

  free(xinit);
  free(xout);

  fftw_free(yf);
  free(mask);

  return 0;
}

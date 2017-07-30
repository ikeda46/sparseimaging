#include <stdio.h>
#include <stdlib.h>
#include "mfista.h"
#include <math.h>
#include <stdio.h>
#include <complex.h> 
#include <fftw3.h>

int *alloc_intvector(int length)
{
    return malloc(sizeof(int)*length);
}

fftw_complex *alloc_fftwvector(int length)
{
    return malloc(sizeof(fftw_complex)*length);
}

int read_intvector(char *fname, int length, int *vector)
{
  FILE *fp;
  int n;

  fp = fopenr(fname);
  n  = fread(vector, sizeof(int), length, fp);
  fclose(fp);

  return(n);
}

int main( void ){
 
  int M = 6590, N = 16384, N1 = 128, N2 = 128, NN = 4, dnum, i,
    *u_idx, *v_idx,
    fftw_sign = FFTW_FORWARD;
  /* fftw_sign = FFTW_BACKWARD; */
  char *u_fname = "data/u_idx.bin", *v_fname = "data/v_idx.bin",
    *noise_fname = "data/noise_std_fft.bin",
    *y_r_fname = "data/vis_r.bin", *y_i_fname = "data/vis_i.bin";
  double *y_r, *y_i, *noise_stdev;
  
  fftw_complex *a, *b, *y_fft;

  printf("\n");
  printf("file for the index of u coordinates:  %s\n", u_fname);
  printf("file for the index of v coordinates:  %s\n", v_fname);
  printf("\n");
  printf("file for noise standard deviations:   %s\n", noise_fname);
  printf("\n");
  printf("file for real values of visibilities: %s\n", y_r_fname);
  printf("file for imag values of visibilities: %s\n", y_i_fname);
  printf("\n");

  u_idx = alloc_intvector(M);
  v_idx = alloc_intvector(M);

  y_r = alloc_vector(M);
  y_i = alloc_vector(M);
  noise_stdev = alloc_vector(M);

  /* read index files */
  
  dnum = read_intvector(u_fname, M, u_idx);
  
  printf("num of u_idx: %d\n",dnum);

  dnum = read_intvector(v_fname, M, v_idx);
  
  printf("num of v_idx: %d\n",dnum);

  printf("u_idx[6590] = %d, v_idx[6590] = %d\n",u_idx[6589],v_idx[6589]);

  /* read visibilities */
  
  dnum = read_V_vector(y_r_fname, M, y_r);
  
  printf("num of real(visibility): %d\n",dnum);
  
  dnum = read_V_vector(y_i_fname, M, y_i);

  printf("num of imag(visibility): %d\n",dnum);

  dnum = read_V_vector(noise_fname, M, noise_stdev);

  printf("num of noise std dev: %d\n",dnum);

  /* y_fft  */

  y_fft = alloc_fftwvector(N1*N2);

  for(i=0;i<M;++i){
    y_fft[(u_idx[i]-1) + N1*(v_idx[i]-1)] = y_r[i] + y_i[i]*I;
  }
  
  
  // a,b は double _Complex 型のC99標準複素配列と実質的に同じ
  // double _Complex a[4] としても動くけど計算速度が低下する可能性あり

  a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NN);
  b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NN);

  // プランの生成
  // フーリエ逆変換つまり位相因子を exp(-k)ではなく exp(+k)とする場合は
  // FFTW_FORWARD の代わりに FFTW_BACKWARD とする
  fftw_plan plan;
  plan = fftw_plan_dft_1d( NN, a, b, fftw_sign, FFTW_ESTIMATE );
 
  // フーリエ変換前の数列値を設定
  a[0] = 1.0 + 0.0*I;
  a[1] = 2.0 + 0.0*I;
  a[2] = 5.0 + 0.0*I;
  a[3] = 3.0 + 0.0*I;
 
  // フーリエ変換実行   b[n]に計算結果が入る
  fftw_execute(plan);
 
  // b[n]の値を表示
  int n;
  for( n=0; n<NN; n++ ){
    printf("b_%d = %+lf %+lf*i\n", n, creal(b[n]), cimag(b[n]) );
  }

  // ここで a[m] の値を変えて再度 fftw_execute(plan) を実行すれば、
  // b[n] が再計算される。

  // 計算終了時、メモリ開放を忘れないように
  if(plan) fftw_destroy_plan(plan);
  fftw_free(a); fftw_free(b);
 
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include "mfista.h"
#include <math.h>
#include <stdio.h>
#include <complex.h> 
#include <fftw3.h>

int *alloc_int_vector(int length)
{
    return malloc(sizeof(int)*length);
}

void fftw_half2full(int N1, int N2, fftw_complex *FT_h, fftw_complex *FT)
{
  int i, j, N1_h, N2_h;

  N1_h = floor(N1/2)+1;
  N2_h = floor(N2/2)+1;

  for(i=0;i<N1;++i)
    for(j=0;j<N2_h;++j)
      FT[N2*i+j] = FT_h[N2_h*i+j];

  for(i=1;i<N1_h-1;++i)
    for(j=1;j<N2_h-1;++j){
      FT[N2*(N1-i)+(N2-j)] = conj(FT_h[N2_h*i+j]);
    }

  for(i=N1_h;i<N1;++i)
    for(j=1;j<N2_h-1;++j){
      FT[N2*(N1-i)+(N2-j)] = conj(FT_h[N2_h*i+j]);
    }

  for(j=1;j<N2_h-1;++j){
    FT[(N2-j)]             = conj(FT_h[j]);
    FT[N2*(N1_h-1)+(N2-j)] = conj(FT_h[N2_h*(N1_h-1)+j]);
  }
}

void fftw_full2half(int N1, int N2, fftw_complex *FT, fftw_complex *FT_h)
{
  int i, j, N2_h;

  N2_h = floor(N2/2)+1;

  for(i=0;i<N1;++i)
    for(j=0;j<N2_h;++j)
      FT_h[N2_h*i+j] = FT[N2*i+j];

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

/*
double calc_F_sqTV_part_fft(int *N, int NX, int NY,
			    fftw_complex *yvec,
			    double *xvec, double lambda_sqtv,
			    int *inc, double *buffvec)
{
  double term1, term2, alpha = -1, beta = 1;

  dcopy_(M, yvec, inc, buffvec, inc);

  dgemv_("N", M, N, &alpha, Amatrix, M, 
	 xvec, inc, &beta, buffvec, inc);

  term1 = ddot_(M, buffvec, inc, buffvec, inc);
  term2 = sqTV(NX, NY, xvec);

  return(term1/2+lambda_sqtv*term2);
}

void dF_dx_fft(int *N, int NX, int NY,
	       fftw_complex *yvec, 
	       double *xvec, double lambda_sqtv,
	       int *inc, double *buffyvec1, double *dFdx)
{
  double alpha = 1, beta;

  calc_yAz(M, N,  yvec, Amatrix, xvec, inc, buffyvec1);

  d_sqTV(NX, NY, xvec, dFdx);

  beta = -lambda_sqtv;

  dgemv_("T", M, N, &alpha, Amatrix, M, buffyvec1, inc, &beta, dFdx, inc);

}
*/
/*
void mfista_L1_TSV_core_fft(fftw_complex *yvec, 
			    int *N, int NX, int NY,
			    double lambda_l1, double lambda_sqtv, double cinit,
			    double *xvec, int nonneg_flag, int looe_flag,
			    struct RESULT *mfista_result)
{
  void (*soft_th)(double *vector, int length, double eta, double *newvec);
  int i, iter, inc = 1;
  double *xtmp, *xnew, *ytmp, *zvec, *dfdx, *yAx,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, costtmp, *cost,
    mu=1, munew, alpha = 1;

  printf("computing image with MFISTA.\n");

  cost  = alloc_vector(MAXITER);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  ytmp  = alloc_vector(*M);
  zvec  = alloc_vector(*N);

  if(nonneg_flag == 0)
    soft_th=soft_threshold;
  else if(nonneg_flag == 1)
    soft_th=soft_threshold_nonneg;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return;
  }

  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  costtmp = calc_F_sqTV_part(M, N, NX, NY,
			     yvec, Amat, zvec, lambda_sqtv, &inc, ytmp);
  l1cost = dasum_(N, zvec, &inc);
  costtmp += lambda_l1*l1cost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    dF_dx(M, N, NX, NY, yvec, Amat, zvec, lambda_sqtv, &inc, ytmp, dfdx);

    Qcore = calc_F_sqTV_part(M, N, NX, NY,
			     yvec, Amat, zvec, lambda_sqtv, &inc, ytmp);

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);
      soft_th(xtmp, *N, lambda_l1/c, xnew);
      Fval = calc_F_sqTV_part(M, N, NX, NY,
			      yvec, Amat, xnew, lambda_sqtv, &inc, ytmp);
      Qval = calc_Q_part(N, xnew, zvec, c, &inc, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);

    Fval += lambda_l1*l1cost;

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

      if((iter>1) && (dasum_(N, xvec, &inc) == 0)) break;
    }

    if((iter>=MINITER) && ((cost[iter-TD]-cost[iter])<EPS)) break;

    mu = munew;
  }
  if(iter == MAXITER){
    printf("%d cost = %f \n",(iter), cost[iter-1]);
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost[iter]);

  printf("\n");

  mfista_result->M = (*M);
  mfista_result->N = (*N);
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->ITER = iter+1;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1 = lambda_l1;
  mfista_result->lambda_sqtv = lambda_sqtv;
  mfista_result->lambda_tv = 0;

  yAx  = alloc_vector(*M);

  calc_yAz(M, N, yvec, Amat, xvec, &inc, yAx);

  mfista_result->sq_error = 0;

  for(i = 0;i< (*M);i++){
    mfista_result->sq_error += yAx[i]*yAx[i];
  }

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

  mfista_result->sqtvcost = sqTV(NX, NY, xvec);
  mfista_result->tvcost = 0;
  mfista_result->finalcost = (mfista_result->sq_error)/2
    + lambda_l1*(mfista_result->l1cost)
    + lambda_sqtv*(mfista_result->sqtvcost);

  free(yAx);

  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(ytmp);
  free(zvec);
}
*/

int main( void ){
 
  int M = 6590, N = 16384, N1 = 128, N2 = 128, N2_h, NN = 4, dnum, i, j,
    *u_idx, *v_idx,
    fftw_sign = FFTW_FORWARD;
  /* fftw_sign = FFTW_BACKWARD; */
  char *u_fname = "data/u_idx.bin", *v_fname = "data/v_idx.bin",
    *noise_fname = "data/noise_std_fft.bin",
    *y_r_fname = "data/vis_r.bin", *y_i_fname = "data/vis_i.bin";
  double *y_r, *y_i, *noise_stdev, *x, *y_r2, *y_i2, *y_r3, *y_i3, *mask_h, *mask;
  
  fftw_complex *a, *b, *y_fft, *y_fft2, *y_fft3, *x_c;
  fftw_plan fftwplan;

  printf("\n");
  printf("file for the index of u coordinates:  %s\n", u_fname);
  printf("file for the index of v coordinates:  %s\n", v_fname);
  printf("\n");
  printf("file for noise standard deviations:   %s\n", noise_fname);
  printf("\n");
  printf("file for real values of visibilities: %s\n", y_r_fname);
  printf("file for imag values of visibilities: %s\n", y_i_fname);
  printf("\n");

  u_idx = alloc_int_vector(M);
  v_idx = alloc_int_vector(M);

  N2_h = floor(N2/2)+1;

  y_r    = alloc_vector(M);
  y_i    = alloc_vector(M);
  y_r2   = alloc_vector(N1*N2_h);
  y_i2   = alloc_vector(N1*N2_h);
  y_r3   = alloc_vector(N1*N2);
  y_i3   = alloc_vector(N1*N2);
  mask_h = alloc_vector(N1*N2_h);
  mask   = alloc_vector(N1*N2);
  noise_stdev = alloc_vector(M);

  x = alloc_vector(N1*N2);

  /* read index files */
  
  dnum = read_int_vector(u_fname, M, u_idx);
  
  printf("num of u_idx: %d\n",dnum);

  dnum = read_int_vector(v_fname, M, v_idx);
  
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

  y_fft  = (fftw_complex*) fftw_malloc(N1*N2*sizeof(fftw_complex));
  y_fft2 = (fftw_complex*) fftw_malloc(N1*N2_h*sizeof(fftw_complex));
  y_fft3 = (fftw_complex*) fftw_malloc(N1*N2*sizeof(fftw_complex));
  x_c    = (fftw_complex*) fftw_malloc(N1*N2*sizeof(fftw_complex));

  for(i=0;i<N1;++i) for(j=0;j<N2;++j) y_fft[N2*i+j] = 0;

  for(i=0;i<M;++i){
    y_fft[N2*(u_idx[i]-1) + (v_idx[i]-1)] = y_r[i]/noise_stdev[i] + y_i[i]*I/noise_stdev[i];
    mask[N2*(u_idx[i]-1) + (v_idx[i]-1)]  = 1/noise_stdev[i];
  }

  fftw_full2half(N1, N2, y_fft, y_fft2);
  for(i=0;i<N1;++i) for(j=0;j<N2_h;++j) mask_h[N2_h*i+j] = mask[N2*i+j];

  for(i = 0;i<10;++i)
    printf("y_fft2[%d] = %+f %+f*i\n",
	   i, creal(y_fft2[i]), cimag(y_fft2[i]) );

  fftw_half2full(N1, N2, y_fft2, y_fft3);

  for(i=0;i<N1;++i){
    for(j=0;j<N2;++j){
      if(creal(y_fft3[N2*i+j])!=creal(y_fft[N2*i+j]))
	printf("y_r[%d][%d]\n",i,j);
      if(cimag(y_fft3[N2*i+j])!=cimag(y_fft[N2*i+j]))
	printf("y_i[%d][%d]\n",i,j);
    }
  }

  for(i = 0;i<N1*N2_h;++i){
    y_r2[i] = creal(y_fft2[i]);
    y_i2[i] = cimag(y_fft2[i]);
  }

  for(i = 0;i<N1*N2;++i){
    y_r3[i] = creal(y_fft3[i]);
    y_i3[i] = cimag(y_fft3[i]);
  }

  write_X_vector("y_r2.out", N1*N2_h, y_r2);
  write_X_vector("y_i2.out", N1*N2_h, y_i2);

  write_X_vector("y_r3.out", N1*N2, y_r3);
  write_X_vector("y_i3.out", N1*N2, y_i3);


  /* fftw */

  fftwplan = fftw_plan_dft_c2r_2d( N1, N2, y_fft2, x, FFTW_ESTIMATE );
  fftw_execute(fftwplan);

  for(i = 0;i<N1*N2;++i){
    x[i] /=(N1*N2);
  }

  for(i = 0;i<10;++i){
    printf("x[%d] = %f\n",i,x[i]);
  }

  write_X_vector("x.out", N1*N2, x);

  fftwplan = fftw_plan_dft_r2c_2d( N1, N2, x, y_fft2, FFTW_ESTIMATE);
  fftw_execute(fftwplan);

  for(i = 0;i<10;++i)
    printf("y_fft2[%d] = %+f %+f*i\n",
	   i, creal(y_fft2[i]), cimag(y_fft2[i]) );
  
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

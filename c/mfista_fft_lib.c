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

void fft_half2full(int NX, int NY, fftw_complex *FT_h, fftw_complex *FT)
{
  int i, j, NX_h, NY_h;
  
  NX_h = (int)floor(((double)NX)/2)+1;
  NY_h = (int)floor(((double)NY)/2)+1;

  for(i=0;i<NX;++i) for(j=0;j<NY_h;++j){
      FT[NY*i+j] = FT_h[NY_h*i+j];
    }

  for(i=1;i<NX_h-1;++i) for(j=1;j<NY_h-1;++j){
      FT[NY*(NX-i)+(NY-j)] = conj(FT_h[NY_h*i+j]);
    }

  for(i=NX_h;i<NX;++i) for(j=1;j<NY_h-1;++j){
      FT[NY*(NX-i)+(NY-j)] = conj(FT_h[NY_h*i+j]);
    }

  for(j=1;j<NY_h-1;++j){
    FT[(NY-j)]             = conj(FT_h[j]);
    FT[NY*(NX_h-1)+(NY-j)] = conj(FT_h[NY_h*(NX_h-1)+j]);
  }
}

void fft_full2half(int NX, int NY, fftw_complex *FT, fftw_complex *FT_h)
{
  int i, j, NY_h;

  NY_h = (int)floor(((double)NY)/2)+1;

  for(i=0;i<NX;++i) for(j=0;j<NY_h;++j) FT_h[NY_h*i+j] = FT[NY*i+j];

}

void calc_yAx_fft(int NX, int NY,
		  fftw_complex *y_fft_h, double *mask, 
		  fftw_complex *yAx_h)
{
  int i, NY_h;
  double sqrtNN;

  NY_h   = (int)floor(((double)NY)/2)+1;
  sqrtNN = sqrt((double)(NX*NY));

  for(i=0;i<NX*NY_h;++i){
    if(mask[i]==0.0)
      yAx_h[i] = 0.0 + 0.0*I;
    else
      yAx_h[i] = y_fft_h[i]-mask[i]*yAx_h[i]/sqrtNN;
  }

}

double norm_fftw_complex(int N, fftw_complex *vec)
{
  int         i;
  double norm_v = 0.0;

  for(i=0;i<N;++i) norm_v += creal(vec[i]*conj(vec[i]));

  return(norm_v);
}

double calc_F_part_fft(int *N, int NX, int NY,
		       fftw_complex *yf_h, double *mask_h,
		       fftw_plan *fftwplan, double *xvec,
		       fftw_complex *yAx_fh, double *x4f, fftw_complex *yAx_f)
{
  double result;
  int inc = 1;

  dcopy_(N, xvec, &inc, x4f, &inc);
  fftw_execute(*fftwplan);

  calc_yAx_fft(NX, NY, yf_h, mask_h, yAx_fh);
  fft_half2full(NX, NY, yAx_fh, yAx_f);

  result = norm_fftw_complex(*N, yAx_f);

  return(result/4);
}

void dF_dx_fft(int *N, int NX, int NY,
	       fftw_complex *yAx_fh, double *mask_h, double *xvec, 
	       fftw_plan *ifftwplan, double *x4f, double *dFdx)
{
  int i, inc = 1, NY_h;
  double sqNN = sqrt((double)(NX*NY));

  NY_h = (int)floor(((double)NY)/2+1);
  
  for(i=0;i<NX*NY_h;++i){
    if(mask_h[i] ==0.0)
      yAx_fh[i] = 0.0 + 0.0*I;
    else
      yAx_fh[i] *= (mask_h[i]/(2*sqNN));
  }

  fftw_execute(*ifftwplan);

  dcopy_(N, x4f, &inc, dFdx, &inc);
}

int mfista_L1_TV_core_fft(fftw_complex *yf, double *mask_h,
			  int *N, int NX, int NY,
			  double lambda_l1, double lambda_tv, double cinit,
			  double *xvec, unsigned int fftw_plan_flag, int nonneg_flag)
{
  double *zvec, *xtmp, *xnew, *dfdx, *ones,
    *rmat, *smat, *npmat, *nqmat, *x4f,
    Qcore, Fval, Qval, c, costtmp, *cost, *pmat, *qmat,
    mu=1, munew, alpha = 1, tmpa, l1cost, tvcost;
  int i, iter, inc = 1, NY_h = ((int)floor(((double)NY)/2)+1);
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan, ifftwplan;

  printf("computing image with MFISTA.\n");

  /* allocate memory space start */ 

  zvec  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  dfdx  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);

  ones  = alloc_vector(*N);
  for(i = 0; i < (*N); ++i) ones[i]=1;

  x4f   = alloc_vector(*N);

  pmat = alloc_matrix(NX-1,NY);
  qmat = alloc_matrix(NX,NY-1);

  npmat = alloc_matrix(NX-1,NY);
  nqmat = alloc_matrix(NX,NY-1);

  rmat = alloc_matrix(NX-1,NY);
  smat = alloc_matrix(NX,NY-1);

  cost  = alloc_vector(MAXITER);

  /* fftw malloc */

  yf_h   = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_f  = (fftw_complex*)  fftw_malloc((*N)*sizeof(fftw_complex));

  /* preparation for fftw */

  fft_full2half(NX, NY, yf, yf_h);

  fftwplan  = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, fftw_plan_flag);
  ifftwplan = fftw_plan_dft_c2r_2d( NX, NY, yAx_fh, x4f, fftw_plan_flag);

  /* initialize xvec */
  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  costtmp = calc_F_part_fft(N, NX, NY, yf_h, mask_h,
			    &fftwplan, xvec, yAx_fh, x4f, yAx_f);

  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;

  tvcost = TV(NX, NY, xvec);
  costtmp += lambda_tv*tvcost;

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0)
      printf("%d cost = %f \n",(iter+1), cost[iter]);

    Qcore = calc_F_part_fft(N, NX, NY, yf_h, mask_h, 
			     &fftwplan, zvec, yAx_fh, x4f, yAx_f);

    dF_dx_fft(N, NX, NY, yAx_fh, mask_h, zvec, &ifftwplan, x4f, dfdx);
            
    for( i = 0; i < MAXITER; i++){

      if(nonneg_flag == 1){
	dcopy_(N, ones, &inc, xtmp, &inc);
	tmpa = -lambda_l1/c;
	dscal_(N, &tmpa, xtmp, &inc);
	tmpa = 1/c;
	daxpy_(N, &tmpa, dfdx, &inc, xtmp, &inc);
	daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

	FGP_nonneg(N, NX, NY, xtmp, lambda_tv/c, FGPITER,
		   pmat, qmat, rmat, smat, npmat, nqmat, xnew);
      }
      else{
	dcopy_(N, dfdx, &inc, xtmp, &inc);
	tmpa = 1/c;
	dscal_(N, &tmpa, xtmp, &inc);
	daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);

	FGP_L1(N, NX, NY, xtmp, lambda_l1/c, lambda_tv/c, FGPITER,
	       pmat, qmat, rmat, smat, npmat, nqmat, xnew);
      }

      Fval = calc_F_part_fft(N, NX, NY, yf_h, mask_h,
			     &fftwplan, xnew, yAx_fh, x4f, yAx_f);

      Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = dasum_(N, xnew, &inc);
    Fval += lambda_l1*l1cost;

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

  /* clear memory */

  if( nonneg_flag == 1 ) free(ones);

  free(npmat);
  free(nqmat);
  free(pmat);
  free(qmat);
  free(rmat);
  free(smat);

  free(dfdx);
  free(cost);
  free(xnew);
  free(xtmp);
  free(zvec);
  free(x4f);

  fftw_free(yAx_fh);
  fftw_free(yf_h);
  fftw_free(yAx_f);

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);
  fftw_cleanup();
  
  return(iter+1);
}


/* TSV */

int mfista_L1_TSV_core_fft(fftw_complex *yf, double *mask_h,
			   int *N, int NX, int NY,
			   double lambda_l1, double lambda_tsv, double cinit,
			   double *xvec, unsigned int fftw_plan_flag, int nonneg_flag)
{
  void (*soft_th)(double *vector, int length, double eta, double *newvec);
  int i, iter, inc = 1, NY_h = ((int)floor(((double)NY)/2)+1);
  double *xtmp, *xnew, *zvec, *dfdx, *x4f,
    Qcore, Fval, Qval, c, cinv, tmpa, l1cost, tsvcost, costtmp, *cost,
    mu=1, munew, alpha = 1, beta = -lambda_tsv;
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan, ifftwplan;

  printf("computing image with MFISTA.\n");

  /* malloc */

  cost  = alloc_vector(MAXITER);
  dfdx  = alloc_vector(*N);
  xnew  = alloc_vector(*N);
  xtmp  = alloc_vector(*N);
  zvec  = alloc_vector(*N);
  x4f   = alloc_vector(*N);

  /* fftw malloc */

  yf_h   = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_f  = (fftw_complex*)  fftw_malloc((*N)*sizeof(fftw_complex));

  /* preparation for fftw */

  fft_full2half(NX, NY, yf, yf_h);

  fftwplan  = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, fftw_plan_flag);
  ifftwplan = fftw_plan_dft_c2r_2d( NX, NY, yAx_fh, x4f, fftw_plan_flag);

  if(nonneg_flag == 0)
    soft_th=soft_threshold;
  else if(nonneg_flag == 1)
    soft_th=soft_threshold_nonneg;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  dcopy_(N, xvec, &inc, zvec, &inc);

  c = cinit;

  costtmp = calc_F_part_fft(N, NX, NY, yf_h, mask_h,
			    &fftwplan, xvec, yAx_fh, x4f, yAx_f);

  l1cost = dasum_(N, xvec, &inc);
  costtmp += lambda_l1*l1cost;

  /* if lambda_tsv >0 */

  if( lambda_tsv > 0 ){
    tsvcost = TSV(NX, NY, xvec);
    costtmp += lambda_tsv*tsvcost;
  }

  /* if lambda_tsv >0 */

  for(iter = 0; iter < MAXITER; iter++){

    cost[iter] = costtmp;

    if((iter % 100) == 0) printf("%d cost = %f \n",(iter+1), cost[iter]);

    Qcore = calc_F_part_fft(N, NX, NY, yf_h, mask_h, 
			    &fftwplan, zvec, yAx_fh, x4f, yAx_f);

    dF_dx_fft(N, NX, NY, yAx_fh, mask_h, zvec, &ifftwplan, x4f, dfdx);

    /* if lambda_tsv >0 */

    if( lambda_tsv > 0 ){
      tsvcost = TSV(NX, NY, zvec);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(NX, NY, zvec, x4f);
      dscal_(N, &beta, x4f, &inc);
      daxpy_(N, &alpha, x4f, &inc, dfdx, &inc);
    }

    /* if lambda_tsv >0 */

    for( i = 0; i < MAXITER; i++){
      dcopy_(N, dfdx, &inc, xtmp, &inc);
      cinv = 1/c;
      dscal_(N, &cinv, xtmp, &inc);
      daxpy_(N, &alpha, zvec, &inc, xtmp, &inc);
      soft_th(xtmp, *N, lambda_l1/c, xnew);

      Fval = calc_F_part_fft(N, NX, NY, yf_h, mask_h,
			     &fftwplan, xnew, yAx_fh, x4f, yAx_f);

      /* if lambda_tsv >0 */

      if( lambda_tsv > 0.0 ){
	tsvcost = TSV(NX, NY, xnew);
	Fval += lambda_tsv*tsvcost;
      }

      /* if lambda_tsv >0 */

      Qval = calc_Q_part(N, xnew, zvec, c, dfdx, xtmp);
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

  /* free */
  
  free(cost);
  free(dfdx);
  free(xnew);
  free(xtmp);
  free(zvec);
  free(x4f);

  fftw_free(yAx_fh);
  fftw_free(yf_h);
  fftw_free(yAx_f);

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);
  fftw_cleanup();

  return(iter+1);
}

void calc_result_fft(fftw_complex *yf, double *mask_h,
		     int M, int NX, int NY,
		     double lambda_l1, double lambda_tv, double lambda_tsv, 
		     double *xvec,
		     struct RESULT *mfista_result)
{
  int i, NY_h = ((int)floor(((double)NY)/2)+1), NN = NX*NY;
  double *x4f, tmpa;
  fftw_complex *yf_h, *yAx_fh, *yAx_f;
  fftw_plan fftwplan;

  /* malloc */

  x4f   = alloc_vector(NN);

  /* fftw malloc */

  yAx_f  = (fftw_complex*)  fftw_malloc((NN)*sizeof(fftw_complex));
  yf_h   = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));
  yAx_fh = (fftw_complex*)  fftw_malloc(NX*NY_h*sizeof(fftw_complex));

  /* preparation for fftw */

  fft_full2half(NX, NY, yf, yf_h);

  fftwplan = fftw_plan_dft_r2c_2d( NX, NY, x4f, yAx_fh, FFTW_ESTIMATE);

  tmpa = calc_F_part_fft(&NN, NX, NY, yf_h, mask_h, &fftwplan, xvec, yAx_fh, x4f, yAx_f);

  mfista_result->sq_error = 2*tmpa;

  /* computing results */

  mfista_result->M  = (int)(M/2);
  mfista_result->N  = NN;
  mfista_result->NX = NX;
  mfista_result->NY = NY;
  mfista_result->maxiter = MAXITER;
	    
  mfista_result->lambda_l1  = lambda_l1;
  mfista_result->lambda_tv  = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;
  
  mfista_result->mean_sq_error = mfista_result->sq_error/((double)M);

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < NN;++i){
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

  /* free */
  
  free(x4f);

  fftw_free(yAx_fh);
  fftw_free(yAx_f);
  fftw_free(yf_h);

  fftw_destroy_plan(fftwplan);
  fftw_cleanup();
}

void mfista_imaging_core_fft(int *u_idx, int *v_idx,
			     double *y_r, double *y_i, double *noise_stdev,
			     int M, int NX, int NY,
			     double lambda_l1, double lambda_tv, double lambda_tsv,
			     double cinit, double *xinit, double *xout,
			     int nonneg_flag, unsigned int fftw_plan_flag,
			     struct RESULT *mfista_result)
{
  int NN, NY_h, i, j, iter, inc = 1;
  double *mask_h, *mask, s_t, e_t;
  fftw_complex *yf;
  struct timespec time_spec1, time_spec2;

  /* set parameters */

  NN = NX*NY;
  NY_h = (int)floor(((double)NY)/2)+1;

  /* allocate vectors */

  mask_h = alloc_vector(NX*NY_h);
  mask   = alloc_vector(NX*NY);
  yf     = (fftw_complex*) fftw_malloc(NX*NY*sizeof(fftw_complex));

  for(i=0;i<NX;++i) for(j=0;j<NY;++j){
      yf[NY*i+j] = 0.0 + 0.0*I;
    }

  for(i=0;i<M;++i){
    yf[ NY*(u_idx[i]) + (v_idx[i]) ] = (y_r[i] + y_i[i]*I)/noise_stdev[i];
    mask[NY*(u_idx[i]) + (v_idx[i])] = 1/noise_stdev[i];
  }

  for(i=0;i<NX;++i)
    for(j=0;j<NY_h;++j){
      mask_h[NY_h*i+j] = mask[NY*i+j];
    }

  dcopy_(&NN, xinit, &inc, xout, &inc);

  /* preparation end */

  get_current_time(&time_spec1);

  /* main loop */

  iter = 0;

  if( lambda_tv == 0 ){

    iter = mfista_L1_TSV_core_fft(yf, mask_h, &NN, NX, NY,
				  lambda_l1, lambda_tsv, cinit, xout,
				  fftw_plan_flag, nonneg_flag);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core_fft(yf, mask_h, &NN, NX, NY,
				 lambda_l1, lambda_tv, cinit, xout,
				 fftw_plan_flag, nonneg_flag);
  }
  else{
    printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
    return;
  }

  get_current_time(&time_spec2);

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER = iter;
  mfista_result->nonneg = nonneg_flag;

  calc_result_fft(yf, mask_h, M, NX, NY,
		  lambda_l1, lambda_tv, lambda_tsv, xout, mfista_result);

  free(mask_h);
  free(mask);
  fftw_free(yf);

  return;
}


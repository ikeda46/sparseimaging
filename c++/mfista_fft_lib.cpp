#include "mfista.hpp"
#include <iomanip>

double fft_half_squareNorm(int Nx, int Ny, fftw_complex *FT_h)
{
  int i, j, Nx_h, Ny_h;
  double sqsum = 0;

  /* set parameters */
  
  Nx_h = (int)floor(((double)Nx)/2) + 1;
  Ny_h = (int)floor(((double)Ny)/2) + 1;

  /* main */

  for(i = 0; i < Nx; i++)
    for(j = 0; j < Ny_h; j++)
      sqsum += pow(FT_h[Ny_h*i+j][0],2.0) + pow(FT_h[Ny_h*i+j][1],2.0);

  for(i = 1; i < Nx_h-1; i++)
    for(j = 1; j < Ny_h-1; j++)
      sqsum += pow(FT_h[Ny_h*i+j][0],2.0) + pow(FT_h[Ny_h*i+j][1],2.0);

  for(i = Nx_h; i < Nx; i++)
    for(j = 1; j < Ny_h-1; j++)
      sqsum += pow(FT_h[Ny_h*i+j][0],2.0) + pow(FT_h[Ny_h*i+j][1],2.0);

  for(j = 1; j < Ny_h-1; j++){
    sqsum += pow(FT_h[j][0],2.0) + pow(FT_h[j][1],2.0);
    sqsum += pow(FT_h[Ny_h*(Nx_h-1)+j][0],2.0) + pow(FT_h[Ny_h*(Nx_h-1)+j][1],2.0);
  }
  return(sqsum);
}

void fft_full2half(int Nx, int Ny, fftw_complex *FT, VectorXcd &FT_h)
{
  int i, j, Ny_h;

  Ny_h = (int)floor(((double)Ny)/2)+1;

  for(i = 0; i < Nx; i++)
    for(j = 0; j < Ny_h; j++)
      FT_h(Ny_h*i + j) = complex<double>(FT[Ny*i + j][0], FT[Ny*i + j][1]);
}

void full2half(int Nx, int Ny, double *mask, VectorXd &mask_h)
{
  int i, j, Ny_h;

  Ny_h = (int)floor(((double)Ny)/2)+1;

  for(i = 0; i < Nx; i++)
    for(j = 0; j < Ny_h; j++)
      mask_h(Ny_h*i + j) = mask[Ny*i + j];
}

void idx2mat(int M, int Nx, int Ny,
	     int *u_idx, int *v_idx, double *vis_r, double *vis_i,
	     double *noise_stdev, fftw_complex *vis, double *mask)
{
  int i, j;

  /* main */
  
  for(i = 0; i < Nx; i++)
    for(j = 0; j < Ny; j++){
      mask[Ny*i + j]   = 0.0;
      vis[Ny*i + j][0] = 0.0;
      vis[Ny*i + j][1] = 0.0;
    }

  for(i = 0; i < M; i++){
    mask[Ny*u_idx[i] + v_idx[i]]   = 1/noise_stdev[i];
    vis[Ny*u_idx[i] + v_idx[i]][0] = vis_r[i]/noise_stdev[i];
    vis[Ny*u_idx[i] + v_idx[i]][1] = vis_i[i]/noise_stdev[i];
  }
}

void calc_yAx_fft(int Nx, int Ny, VectorXcd &y_fft_h, VectorXd &mask_h, fftw_complex *cvec)
{
  int i, Ny_h;
  double sqrtNN;

  /* set parameters */

  Ny_h   = (int)floor(((double)Ny)/2) + 1;
  sqrtNN = sqrt((double)(Nx*Ny));

  /* main */

  for(i = 0; i < Nx*Ny_h; i++){
    cvec[i][0] = y_fft_h(i).real() - mask_h(i)*cvec[i][0]/sqrtNN;
    cvec[i][1] = y_fft_h(i).imag() - mask_h(i)*cvec[i][1]/sqrtNN;
  }
}

double calc_F_part_fft(int Nx, int Ny,
		       VectorXcd &vis_h, VectorXd &mask_h,
		       fftw_plan *fftwplan, VectorXd &xvec,
		       fftw_complex *cvec, double *rvec)
{
  int i, NN = Nx* Ny;

  for(i = 0; i < NN; i++) rvec[i] = xvec(i);

  fftw_execute(*fftwplan);
  calc_yAx_fft(Nx, Ny, vis_h, mask_h, cvec);
  return(fft_half_squareNorm(Nx, Ny, cvec)/4);
}

void dF_dx_fft(VectorXd &dfdx, int Nx, int Ny,
	       fftw_complex *yAx_fh, VectorXd &mask_h, VectorXd &xvec, 
	       fftw_plan *ifftwplan, double *rvec)
{
  int i, Ny_h, NN = Nx*Ny;
  double sqNN = sqrt((double)(Nx*Ny));

  Ny_h = (int)floor(((double)Ny)/2) + 1;
  
  for(i=0;i<Nx*Ny_h;++i){
    yAx_fh[i][0] *= (mask_h(i)/(2*sqNN));
    yAx_fh[i][1] *= (mask_h(i)/(2*sqNN));
  }

  fftw_execute(*ifftwplan);

  for(i = 0; i < NN; i++) dfdx(i) = rvec[i];
}

/* TSV */

int mfista_L1_TSV_core_fft(int Nx, int Ny, int maxiter, double eps,
			   fftw_complex *vis, double *mask,
			   double lambda_l1, double lambda_tsv,
			   double *cinit, double *xinit, double *xout,
			   int nonneg_flag, unsigned int fftw_plan_flag,
			   int box_flag, float *cl_box)
{
  void (*soft_th_box)(VectorXd &newvec, VectorXd &vector, double eta, int box_flag, VectorXd &box);
  int NN = Nx*Ny, i, iter, Ny_h;
  double *rvec, 
    Qcore, Fval, Qval, c, tmpa, tmpb, l1cost, tsvcost, costtmp,
    mu=1, munew;
  fftw_complex *cvec;
  fftw_plan fftwplan, ifftwplan;

  VectorXd cost, xtmp, xnew, zvec, dfdx, dtmp, xvec, box, mask_h, buf_diff;
  VectorXcd yAx_h, vis_h;

  /* set parameters */

  Ny_h = ((int)floor(((double)Ny)/2)+1);

  cout << "computing image with MFISTA." << endl;
  cout << "stop if iter = " << maxiter << ", or Delta_cost < " << eps << endl;

  /* allocate variables */

  cost   = VectorXd::Zero(maxiter);
  dfdx   = VectorXd::Zero(NN);
  xnew   = VectorXd::Zero(NN);
  xtmp   = VectorXd::Zero(NN);
  dtmp   = VectorXd::Zero(NN);
  box    = VectorXd::Zero(NN);

  vis_h   = VectorXcd::Zero(Nx*Ny_h);

  buf_diff = VectorXd::Zero(Nx-1);

  xvec = Map<VectorXd>(xinit,NN);
  zvec = xvec;

  mask_h = VectorXd::Zero(Nx*Ny_h);
 
  /* fftw malloc */

  rvec  = new double [4*NN];
  cvec = (fftw_complex*) fftw_malloc(Nx*Ny_h*sizeof(fftw_complex));

  /* preparation for fftw */

  full2half(Nx, Ny, mask, mask_h);
  fft_full2half(Nx, Ny, vis, vis_h);

#ifdef PTHREAD
  int omp_num = THREAD_NUM;
  cout << "Run mfista with " << omp_num << " threads." << endl;

  if(fftw_init_threads()==0)
    cout << "Could not initialize multi threads for fftw3." << endl;
#endif
  
  fftwplan  = fftw_plan_dft_r2c_2d( Nx, Ny, rvec, cvec, fftw_plan_flag);
  ifftwplan = fftw_plan_dft_c2r_2d( Nx, Ny, cvec, rvec, fftw_plan_flag);

  /* initialization */

  if(nonneg_flag == 0)
    soft_th_box =soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box =soft_threshold_nonneg_box;
  else {
    cout << "nonneg_flag must be chosen properly." << endl;
    return(0);
  }

  c = *cinit;

  /* main */

  costtmp = calc_F_part_fft(Nx, Ny, vis_h, mask_h,
			    &fftwplan, xvec, cvec, rvec);

  // from OK 
  l1cost = xvec.lpNorm<1>();
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(Nx, Ny, xvec, buf_diff);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost(iter) = costtmp;

    if((iter % 100) == 0){
      cout << iter+1 << " cost = " << fixed << setprecision(5)
	   << cost(iter) << ", c = " << c << endl;
    }

    Qcore = calc_F_part_fft(Nx, Ny, vis_h, mask_h, 
			    &fftwplan, zvec, cvec, rvec);

    dF_dx_fft(dfdx, Nx, Ny, cvec, mask_h, zvec, &ifftwplan, rvec);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(Nx, Ny, zvec, buf_diff);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(dtmp, Nx, Ny, zvec);
      dfdx.array() -= lambda_tsv*dtmp.array();
    }

    for( i = 0; i < maxiter; i++){
      xtmp.array() = zvec.array() + dfdx.array()/c;
      soft_th_box(xnew, xtmp, lambda_l1/c, box_flag, box);
      
      Fval = calc_F_part_fft(Nx, Ny, vis_h, mask_h,
			     &fftwplan, xnew, cvec, rvec);

      if( lambda_tsv > 0.0 ){
	tsvcost = TSV(Nx, Ny, xnew, buf_diff);
	Fval += lambda_tsv*tsvcost;
      }

      Qval = calc_Q_part(xnew, zvec, c, dfdx, xtmp);
      Qval += Qcore;

      if(Fval<=Qval) break;

      c *= ETA;
    }

    c /= ETA;

    munew = (1+sqrt(1+4*mu*mu))/2;

    l1cost = xnew.lpNorm<1>();
    Fval += lambda_l1*l1cost;

    zvec = xvec;

    if(Fval < cost(iter)){

      costtmp = Fval;

      tmpa = 1+((mu-1)/munew);
      tmpb = ((1-mu)/munew);

      zvec.array() = tmpa * xnew.array() + tmpb * zvec.array();

      xvec = xnew;
    }
    else{
      tmpa = mu/munew;
      tmpb = 1-(mu/munew);

      zvec.array() = tmpa * xnew.array() + tmpb * zvec.array();

      // another stopping rule 
      if((iter>1) && (xvec.lpNorm<1>() == 0)) break;
    }

    if((iter>=MINITER) && ((cost(iter-TD)-cost(iter))< eps )) break;

    mu = munew;
  }
  if(iter == maxiter){
    cout << iter << " cost = " << cost(iter-1) << endl;
    iter = iter -1;
  }
  else
    cout << iter+1 << " cost = "  << cost(iter) << endl;

  cout << endl;

  *cinit = c;

  for(i = 0; i < NN; i++) xout[i] = xvec(i);

  /* free */
  
  delete cvec;
  delete rvec;

  fftw_destroy_plan(fftwplan);
  fftw_destroy_plan(ifftwplan);

#ifdef PTHREAD
  fftw_cleanup_threads();
#else
  fftw_cleanup();
#endif

  cout << resetiosflags(ios_base::floatfield);
  
  return(iter+1);
  // to OK
}

/* results */

void calc_result_fft(int M, int Nx, int Ny,
		     fftw_complex *vis, double *mask,
		     double lambda_l1, double lambda_tv, double lambda_tsv, 
		     double *x,
		     struct RESULT *mfista_result)
{
  int i, Ny_h = ((int)floor(((double)Ny)/2)+1), NN = Nx*Ny;
  double *rvec, tmp;
  fftw_complex *cvec;
  fftw_plan fftwplan;

  VectorXd mask_h, buf_diff, xvec;
  VectorXcd vis_h;

  /* allocate variables */

  rvec   = new double [NN];

  xvec   = Map<VectorXd>(x,NN);
  mask_h = VectorXd::Zero(Nx*Ny_h);
  vis_h  = VectorXcd::Zero(Nx*Ny_h);
  
  /* fftw malloc */

  cvec = (fftw_complex*) fftw_malloc(Nx*Ny_h*sizeof(fftw_complex));

  /* preparation for fftw */

  full2half(Nx, Ny, mask, mask_h);
  fft_full2half(Nx, Ny, vis, vis_h);

  fftwplan = fftw_plan_dft_r2c_2d( Nx, Ny, rvec, cvec, FFTW_ESTIMATE);

  /* computing results */
  
  tmp = calc_F_part_fft(Nx, Ny, vis_h, mask_h,
			&fftwplan, xvec, cvec, rvec);

  /* saving results */

  mfista_result->sq_error = 2*tmp;

  mfista_result->M  = (int)(M/2);
  mfista_result->N  = NN;
  mfista_result->NX = Nx;
  mfista_result->NY = Ny;
	    
  mfista_result->lambda_l1  = lambda_l1;
  mfista_result->lambda_tv  = lambda_tv;
  mfista_result->lambda_tsv = lambda_tsv;
  
  mfista_result->mean_sq_error = mfista_result->sq_error/((double)M);

  mfista_result->l1cost   = 0;
  mfista_result->N_active = 0;

  for(i = 0;i < NN;++i){
    tmp = fabs(xvec[i]);
    if(tmp > 0){
      mfista_result->l1cost += tmp;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->finalcost = (mfista_result->sq_error)/2;

  if(lambda_l1 > 0)
    mfista_result->finalcost += lambda_l1*(mfista_result->l1cost);

  if(lambda_tsv > 0){
    mfista_result->tsvcost = TSV(Nx, Ny, xvec, buf_diff);
    mfista_result->finalcost += lambda_tsv*(mfista_result->tsvcost);
  }

  /* free */
  
  delete cvec;
  delete rvec;

  fftw_destroy_plan(fftwplan);
  fftw_cleanup();
}

/* main subroutine */

void mfista_imaging_core_fft(int *u_idx, int *v_idx, 
			     double *y_r, double *y_i, double *noise_stdev,
			     int M, int Nx, int Ny, int maxiter, double eps,
			     double lambda_l1, double lambda_tv, double lambda_tsv,
			     double cinit, double *xinit, double *xout,
			     int nonneg_flag, unsigned int fftw_plan_flag,
			     int box_flag, float *cl_box,
			     struct RESULT *mfista_result)
{
  int i, iter = 0;
  double epsilon, *mask, s_t, e_t, c = cinit;
  struct timespec time_spec1, time_spec2;
  fftw_complex *vis;

  for(epsilon=0, i=0;i<M;++i) epsilon += y_r[i]*y_r[i] + y_i[i]*y_i[i];
    
  epsilon *= eps/((double)M);

  vis   = (fftw_complex*) fftw_malloc(Nx*Ny*sizeof(fftw_complex));
  mask = new double [Nx*Ny];

  idx2mat(M, Nx, Ny, u_idx, v_idx, y_r, y_i, noise_stdev, vis, mask);

  get_current_time(&time_spec1);

  if( lambda_tv == 0 ){
    iter = mfista_L1_TSV_core_fft(Nx, Ny, maxiter, epsilon,
				  vis, mask, lambda_l1, lambda_tsv, &c, xinit, xout,
				  nonneg_flag, fftw_plan_flag, box_flag, cl_box);
  }
  // else if( lambda_tv != 0  && lambda_tsv == 0 ){
  //   iter = mfista_L1_TV_core_fft(Nx, Ny, maxiter, epsilon,
  // 				 vis, mask, lambda_l1, lambda_tv, &c, xinit, xout,
  // 				 nonneg_flag, fftw_plan_flag, box_flag, cl_box);
  // }
  else{
    cout << "We have not implemented TV option." << endl;
    // cout << "You cannot set both of lambda_TV and lambda_TSV positive." << endl;
    return;
  }

  get_current_time(&time_spec2);

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER      = iter;
  mfista_result->nonneg    = nonneg_flag;
  mfista_result->Lip_const = c;
  mfista_result->maxiter   = maxiter;

  calc_result_fft(M, Nx, Ny, vis, mask, lambda_l1, lambda_tv, lambda_tsv, xout, mfista_result);

  delete vis;
  delete mask;

}


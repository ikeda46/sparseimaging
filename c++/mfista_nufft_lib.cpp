#include "mfista.hpp"

// mfist_nufft_lib

int idx_fftw(int m, int Mr)
{
  if(m >= 0 && m < Mr/2) return(m);
  else if(m >= -Mr/2)    return(m+Mr);
  else                   return(2*Mr);
}

int m2mr(int id, int N)
{
  if(id < N/2) return(id + 3*N/2);
  else         return(id - N/2);
}

void preNUFFT(VectorXd &u, VectorXd &v,
	      VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y,
	      MatrixXd &E4mat, VectorXi &mx, VectorXi &my)
{
  int i, j, k, M, Nx, Ny;
  double taux, tauy, coeff, xix, xiy, Mrx, Mry, tmp3x, tmp3y,
    tmpx, tmpy, pi, tmpi, tmpj, tmpcoef[2*MSP];

  pi = M_PI;

  M  = E1.size();
  Nx = E4mat.rows();
  Ny = E4mat.cols();

  Mrx = (double) 2*Nx;
  Mry = (double) 2*Ny;

  taux = 12/((double)(Nx*Nx));
  tauy = 12/((double)(Ny*Ny));

  coeff = pi/sqrt(taux*tauy);

  tmp3x = pow((pi/Mrx),2.0)/taux;
  tmp3y = pow((pi/Mry),2.0)/tauy;

  for(j = 0; j < 2*MSP; j++) tmpcoef[j] = (double)(-MSP+1+j);

  for(k = 0; k < M; k++){

    tmpx = round(u(k)*Mrx/(2*pi));
    tmpy = round(v(k)*Mry/(2*pi));
    
    mx(k) = (int)tmpx;
    my(k) = (int)tmpy;

    xix = (2*pi*tmpx)/Mrx;
    xiy = (2*pi*tmpy)/Mry;

    /* E1 */

    tmpx = -pow((u(k)-xix),2.0)/(4*taux);
    tmpy = -pow((v(k)-xiy),2.0)/(4*tauy);

    E1(k) = exp( tmpx + tmpy );

    /* E2 */

    tmpx = pi*(u(k)-xix)/(Mrx*taux);
    tmpy = pi*(v(k)-xiy)/(Mry*tauy);
    
    for(j = 0; j < 2*MSP; j++){
      //      E2x(k, j) = exp(tmpcoef[j]*tmpx);
      //      E2y(k, j) = exp(tmpcoef[j]*tmpy);
      E2x(k, j) = exp(tmpcoef[j]*tmpx-tmp3x*((double)((j-MSP+1)*(j-MSP+1))));
      E2y(k, j) = exp(tmpcoef[j]*tmpy-tmp3y*((double)((j-MSP+1)*(j-MSP+1))));
    }
  }

  for(i = 0; i < Nx; i++){

    tmpi = (double)(i-Nx/2);
    tmpx = taux*tmpi*tmpi;

    for(j = 0; j< Ny; j++){
      tmpj = (double)(j-Nx/2);
      tmpy = tauy*tmpj*tmpj;
      E4mat(i,j) = coeff*exp(tmpx + tmpy);
    }
  }
}

void NUFFT2d1(VectorXd &Xout,
	      VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y,
	      MatrixXd &E4mat, VectorXi &mx, VectorXi &my,
 	      fftw_complex *in, double *out, fftw_plan *fftwplan_c2r,
 	      VectorXcd &Fin)
{
  int M, Nx, Ny, Mrx, Mry, Mh, j, k, lx, ly, idx, idy, sign;
  double MM;
  complex<double> v0, vy, tmpc;
    
  M =  E1.size();
  Nx = E4mat.rows();
  Ny = E4mat.cols();

  Mrx = 2*Nx;
  Mh =  Ny + 1;
  Mry = 2*Ny;
  MM  = (double)(Mrx*Mry);

  for(j = 0; j < Mrx*Mh; j++){
    in[j][0] = 0;
    in[j][1] = 0;
  }

  for(k = 0; k < M; k++){

    if(NU_SIGN == 1) v0 = E1(k)*Fin(k);
    else             v0 = E1(k)*conj(Fin(k));

    // for original

    for(ly = 0; ly < 2*MSP; ly++){

      vy = v0*E2y(k, ly)/2.0;

      for(sign = -1; sign < 2; sign +=2){
	idy = idx_fftw(sign*(my(k)+ly-MSP+1),Mry);

	if(idy < (Ny+1))
	  for(lx = 0; lx < 2*MSP; lx++){
	    idx = idx_fftw(sign*(mx(k)+lx-MSP+1),Mrx);
	    tmpc = (complex<double>) vy*E2x(k,lx);
	    in[idx*Mh + idy][0] += real(tmpc);
	    in[idx*Mh + idy][1] += sign*imag(tmpc);
	  }
      }
    }
  }

  fftw_execute(*fftwplan_c2r);

  for(k = 0; k < Nx; k++){
    idx = m2mr(k,Nx);
    for(j = 0; j < Ny; j++){
      idy = m2mr(j,Ny);
      Xout(k*Ny + j) = out[idx*Mry + idy]*E4mat(k,j)/MM;
    }
  }
}

void NUFFT2d2(VectorXcd &Fout, VectorXd &E1,
	      MatrixXd &E2x, MatrixXd &E2y,
 	      MatrixXd &E4mat, VectorXi &mx, VectorXi &my,
 	      double *in, fftw_complex *out, fftw_plan *fftwplan_r2c,
 	      VectorXd &Xin)
{
  int M, Nx, Ny, Mrx, Mry, Mh, i, j, k, lx, ly, idx, idy, sign;
  double tmp, MM, f_r;

  M =  E1.size();
  Nx = E4mat.rows();
  Ny = E4mat.cols();

  Mrx = 2*Nx;
  Mry = 2*Ny;
  Mh  = Ny+1;
  MM = (double)Mrx*Mry;

  Fout = VectorXcd::Zero(M);

  for(i = 0; i < 2*Nx; i++) for(j = 0; j < 2*Ny; j++) in[i*Mry + j] = 0;

  for(i = 0; i < Nx; i++){
    idx = m2mr(i,Nx);
    for(j = 0; j < Ny; j++){
      idy = m2mr(j,Ny);
      in[idx*Mry + idy] = Xin(i*Ny + j)*E4mat(i,j);
    }
  }

  fftw_execute(*fftwplan_r2c);

  for(k = 0; k < M; k++){
    for(ly = 0; ly < 2*MSP; ly++){

      idy = -1;
      
      j = idx_fftw((my(k)+ly-MSP+1),Mry);
      if(j < (Ny+1)){
	idy = j;
	sign = 1;
      }
      else{
	j = idx_fftw(-(my(k)+ly-MSP+1),Mry);
	if(j < (Ny+1)){
	  idy = j;
	  sign = -1;
	}
      }

      if( idy >= 0){
	f_r = E1(k)*E2y(k,ly)/MM;
	for(lx = 0; lx < 2*MSP; lx++){

	  idx = idx_fftw(sign*(mx(k)+lx-MSP+1),Mrx);
	  tmp = f_r*E2x(k,lx);
	  
	  Fout(k) += complex<double>
	    (out[idx*Mh + idy][0], sign*(NU_SIGN)*out[idx*Mh + idy][1])*tmp;
	}
      }
    }
  }
}

double calc_F_part_nufft(VectorXcd &yAx,
			 VectorXd &E1,
			 MatrixXd &E2x, MatrixXd &E2y,
			 MatrixXd &E4mat, VectorXi &mx, VectorXi &my,
 			 double *in_r, fftw_complex *out_c,
			 fftw_plan *fftwplan_r2c,
 			 VectorXcd &vis, VectorXd &weight, VectorXd &xvec)
{
  NUFFT2d2(yAx, E1, E2x, E2y, E4mat, mx, my,
 	   in_r, out_c, fftwplan_r2c, xvec);

  yAx = (vis.array() - yAx.array())*weight.array();

  return(yAx.squaredNorm()/2);
}

void dF_dx_nufft(VectorXd &dFdx,
		 VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y,
		 MatrixXd &E4mat,
		 VectorXi &mx, VectorXi &my,			 
		 fftw_complex *in_c, double *out_r, fftw_plan *fftwplan_c2r,
		 VectorXd &weight, VectorXcd &yAx)
{
  yAx.array() *= weight.array();

  NUFFT2d1(dFdx, E1, E2x, E2y, E4mat, mx, my,
	   in_c, out_r, fftwplan_c2r, yAx);
}

/* TSV */

int mfista_L1_TSV_core_nufft(double *xout,
			     int M, int Nx, int Ny, double *u_dx, double *v_dy,
			     int maxiter, double eps,
			     double *vis_r, double *vis_i, double *vis_std,
			     double lambda_l1, double lambda_tsv,
			     double *cinit, double *xinit, 
			     int nonneg_flag, int box_flag, float *cl_box)
{
  void (*soft_th_box)(VectorXd &newvec, VectorXd &vector, double eta, int box_flag, VectorXd &box);
  
  int NN = Nx*Ny, MMh = 2*Nx*(Ny+1), i, iter;
  double Qcore, Fval, Qval, c, tmpa, tmpb, l1cost, tsvcost, costtmp, 
    mu=1, munew, *rvec;
  fftw_complex *cvec;
  fftw_plan fftwplan_c2r, fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  VectorXi mx, my;
  VectorXd E1, cost, xtmp, xnew, zvec, dfdx, dtmp, weight, xvec, box, u, v, buf_diff;
  VectorXcd yAx, vis;
  MatrixXd E2x, E2y, E4mat;

  cout << "Memory allocation and preparations." << endl << endl;
  
  mx    = VectorXi::Zero(M);
  my    = VectorXi::Zero(M);
  E1    = VectorXd::Zero(M);
  E2x   = MatrixXd::Zero(M,2*MSP);
  E2y   = MatrixXd::Zero(M,2*MSP);
  E4mat = MatrixXd::Zero(Nx,Ny);

  cost   = VectorXd::Zero(maxiter);
  dfdx   = VectorXd::Zero(NN);
  xnew   = VectorXd::Zero(NN);
  xtmp   = VectorXd::Zero(NN);
  xvec   = VectorXd::Zero(NN);
  zvec   = VectorXd::Zero(NN);
  dtmp   = VectorXd::Zero(NN);
  box    = VectorXd::Zero(NN);

  yAx    = VectorXcd::Zero(M);
  vis    = VectorXcd::Zero(M);
  weight = VectorXd::Zero(M);

  buf_diff = VectorXd::Zero(Nx-1);

  u = Map<VectorXd>(u_dx,M);
  v = Map<VectorXd>(v_dy,M);
  xvec = Map<VectorXd>(xinit,NN);
  zvec = xvec;

  if(box_flag == 1){
    for(i = 0; i < NN; i++) box(i) = (double)cl_box[i];
  }
  else box = VectorXd::Zero(NN);

  for(i = 0; i < M; i++){
    vis(i) = complex<double>(vis_r[i],vis_i[i]);
    weight(i) = 1/vis_std[i];
  }

  cout << "Preparation for FFT." << endl; 

  // prepare for nufft

  preNUFFT(u, v, E1, E2x, E2y, E4mat, mx, my);

  // for fftw
  
  rvec  = new double [4*NN];
  cvec  = (fftw_complex*) fftw_malloc(MMh*sizeof(fftw_complex));

  for(i = 0; i< MMh; i++) {cvec[i][0]=0;cvec[i][1]=0;}
  for(i = 0; i< 4*NN; i++){rvec[i]=0;}

  fftwplan_c2r = fftw_plan_dft_c2r_2d(2*Nx,2*Ny, cvec, rvec, fftw_plan_flag);
  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvec, cvec, fftw_plan_flag);
  fftw_execute(fftwplan_r2c);
  fftw_execute(fftwplan_c2r);
  
  // initialization

  if(nonneg_flag == 0)
    soft_th_box =soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box =soft_threshold_nonneg_box;
  else {
    printf("nonneg_flag must be chosen properly.\n");
    return(0);
  }

  cout << "Done." << endl; 

  c = *cinit;

  cout << "computing image with MFISTA with NUFFT." << endl;
  cout << "stop if iter = " << maxiter << " or Delta_cost < " << eps << endl;

  // main

  costtmp = calc_F_part_nufft(yAx, E1, E2x, E2y, E4mat, mx, my,
			      rvec, cvec, &fftwplan_r2c, vis, weight, xvec);

  l1cost = xvec.lpNorm<1>();
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(Nx, Ny, xvec, buf_diff);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost(iter) = costtmp;

    if((iter % 10) == 0)
      printf("%d cost = %f, c = %f \n",(iter+1), cost(iter), c);

    Qcore = calc_F_part_nufft(yAx, E1, E2x, E2y, E4mat, mx, my,
 			      rvec, cvec, &fftwplan_r2c, vis, weight, zvec);

    dF_dx_nufft(dfdx, E1, E2x, E2y, E4mat, mx, my,
 		cvec, rvec, &fftwplan_c2r, weight, yAx);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(Nx, Ny, zvec, buf_diff);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(dtmp, Nx, Ny, zvec);
      dfdx.array() -= lambda_tsv*dtmp.array();
    }

    for(i = 0; i < maxiter; i++){
      xtmp.array() = zvec.array() + dfdx.array()/c;
      soft_th_box(xnew, xtmp, lambda_l1/c, box_flag, box);

      Fval = calc_F_part_nufft(yAx, E1, E2x, E2y, E4mat, mx, my,
 			       rvec, cvec, &fftwplan_r2c, vis, weight, xnew);

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
    printf("%d cost = %f \n",(iter), cost(iter-1));
    iter = iter -1;
  }
  else
    printf("%d cost = %f \n",(iter+1), cost(iter));

  printf("\n");

  *cinit = c;

  for(i = 0; i < NN; i++) xout[i] = xvec(i);

  // free memory
  
  delete cvec;
  delete rvec;

  fftw_destroy_plan(fftwplan_c2r);
  fftw_destroy_plan(fftwplan_r2c);

  return(iter+1);
}

/* results */

void calc_result_nufft(struct RESULT *mfista_result,
		       int M, int Nx, int Ny, double *u_dx, double *v_dy,
		       double *vis_r, double *vis_i, double *vis_std,
		       double lambda_l1, double lambda_tv, double lambda_tsv, 
		       double *xvec)
{
  int i, NN = Nx*Ny;
  double tmp, *rvec;
  fftw_complex *cvec;
  fftw_plan fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  VectorXi mx, my;
  VectorXd E1, weight, x, u, v, buf_diff;
  VectorXcd yAx, vis;
  MatrixXd E2x, E2y, E4mat;

  cout << "Memory allocation and preparations." << endl; 
  
  mx    = VectorXi::Zero(M);
  my    = VectorXi::Zero(M);
  E1    = VectorXd::Zero(M);
  E2x   = MatrixXd::Zero(M,2*MSP);
  E2y   = MatrixXd::Zero(M,2*MSP);
  E4mat = MatrixXd::Zero(Nx,Ny);

  buf_diff = VectorXd::Zero(Nx-1);
  u = Map<VectorXd>(u_dx,M);
  v = Map<VectorXd>(v_dy,M);
  x = Map<VectorXd>(xvec,NN);

  // prepare for nufft

  preNUFFT(u, v, E1, E2x, E2y, E4mat, mx, my);

  // for fftw

  cvec  = (fftw_complex*) fftw_malloc(2*Nx*(Ny+1)*sizeof(fftw_complex));
  rvec  = new double [4*NN];

  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvec, cvec, fftw_plan_flag);

  // complex malloc

  yAx    = VectorXcd::Zero(M);
  vis    = VectorXcd::Zero(M);
  weight = VectorXd::Zero(M);

  for(i = 0; i < M; i++){
    vis(i) = complex<double>(vis_r[i],vis_i[i]);
    weight(i) = 1/vis_std[i];
  }

  // computing results
  
  tmp = calc_F_part_nufft(yAx, E1, E2x, E2y, E4mat, mx, my,
 			  rvec, cvec, &fftwplan_r2c, vis, weight, x);

//   /* saving results */

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

  for(i = 0; i < NN; i++){
    tmp = fabs(x(i));
    if(tmp > 0){
      mfista_result->l1cost += tmp;
      ++ mfista_result->N_active;
    }
  }

  mfista_result->finalcost = (mfista_result->sq_error)/2;

  if(lambda_l1 > 0)
    mfista_result->finalcost += lambda_l1*(mfista_result->l1cost);

  if(lambda_tsv > 0){
    mfista_result->tsvcost = TSV(Nx, Ny, x, buf_diff);
    mfista_result->finalcost += lambda_tsv*(mfista_result->tsvcost);
  }
  //  else if (lambda_tv > 0){
  //    mfista_result->tvcost = TV(Nx, Ny, x);
  //    mfista_result->finalcost += lambda_tv*(mfista_result->tvcost);
  //  }

  // free 

  delete cvec;
  delete rvec;

  fftw_destroy_plan(fftwplan_r2c);

}

/* main subroutine */

void mfista_imaging_core_nufft(double *u_dx, double *v_dy, 
			       double *vis_r, double *vis_i, double *vis_std,
			       int M, int Nx, int Ny, int maxiter, double eps,
			       double lambda_l1, double lambda_tv, double lambda_tsv,
			       double cinit, double *xinit, double *xout,
			       int nonneg_flag, int box_flag, float *cl_box,
			       struct RESULT *mfista_result)
{
  int i, iter = 0;
  double epsilon, s_t, e_t, c = cinit;
  struct timespec time_spec1, time_spec2;

  // start main part */

  epsilon = 0;
  for(i = 0; i < M; i++) epsilon += vis_r[i]*vis_r[i] + vis_i[i]*vis_i[i];

  epsilon *= eps/((double)M);

  get_current_time(&time_spec1);

  if( lambda_tv == 0 ){
    iter = mfista_L1_TSV_core_nufft(xout, M, Nx, Ny, u_dx, v_dy, maxiter, epsilon, vis_r, vis_i, vis_std,
				    lambda_l1, lambda_tsv, &c, xinit, nonneg_flag, box_flag, cl_box);
  }
  //  else if( lambda_tv != 0  && lambda_tsv == 0 ){
  //    iter = mfista_L1_TV_core_nufft(xout, M, Nx, Ny, u_dx, v_dy, maxiter, epsilon, vis_r, vis_i, vis_std,
  //                                   lambda_l1, lambda_tv, &c, xinit, nonneg_flag, box_flag, cl_box);
  //}
  else{
    cout << "We have not implemented TV option." << endl;
    // printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
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

  calc_result_nufft(mfista_result, M, Nx, Ny, u_dx, v_dy, vis_r, vis_i, vis_std,
		    lambda_l1, lambda_tv, lambda_tsv, xout);

}

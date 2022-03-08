#include "selfcal.hpp"

int adjust_g_st(int Gnum, int Stnum, int **gid2vid_tbl, VectorXd &ainv, VectorXd &z)
{
  int      i, st;
  VectorXd mask, eta, z_tmp, ainv_tmp, gst_num, zsum;

  eta      = VectorXd::Zero(Stnum);
  z_tmp    = VectorXd::Zero(Stnum);
  ainv_tmp = VectorXd::Zero(Stnum);
  gst_num  = VectorXd::Zero(Stnum);
  zsum     = VectorXd::Zero(Stnum);
  mask     = VectorXd::Ones(Gnum);

  for(i = 0; i<Gnum; i++){
    st            = gid2vid_tbl[i][0];
    z_tmp(st)    +=    z[i];
    ainv_tmp(st) += ainv[i];
    gst_num(st)  +=     1.0;
  }

  eta.array() = (z_tmp.array() - gst_num.array())/ainv_tmp.array();

  for(i = 0; i<Gnum; i++){
    st    = gid2vid_tbl[i][0];
    z(i) -= eta(st)*ainv(i);
  }

  for(i = 0; i<Gnum; i++){
    if(z(i)<0){
      mask(i) = 0.0;
    }
    else{
      st    = gid2vid_tbl[i][0];
      zsum(st) += z(i);
    }
  }

  z.array() *= mask.array();

  for(i = 0; i<Gnum; i++){
      st    = gid2vid_tbl[i][0];
      z(i) *= gst_num(st)/zsum(st);
  }

  return((int)round(mask.sum()));
}

int adjust_g(int Gnum, VectorXd &ainv, VectorXd &z)
{
  int i;
  double eta, zmin;
  VectorXd mask;

  mask = VectorXd::Ones(Gnum);

  eta = (z.sum()-Gnum)/(ainv.sum());

  zmin = z.minCoeff();

  while(zmin < eta){

    for(i = 0;i<Gnum;i++) if( z(i) < eta ) mask(i) = 0.0;

    z.array() *= mask.array();

    zmin = z.maxCoeff();
    for(i = 0; i < Gnum; ++i)
      if(z(i) < zmin && z(i) > 0.0)
        zmin = z(i);

    eta = (z.sum()-Gnum)/ainv.sum();

  }

  z -= eta*ainv;
  z.array() *= mask.array();

  return((int)round(mask.sum()));
}

double cost_gh(VectorXcd &wvis, VectorXcd &wy, VectorXd &sig,
                VectorXcd &g, VectorXcd &h,
                int **gid2vid_tbl, int **vid2gid_st, double *time_tbl,
                int M, int Gnum,
                double lambda_1, double lambda_2, double rho, int prt)
{
  int i, gid1, gid2, st, k;
  double d_t, w_at, cost, tmpd0, tmpd1;
  complex<double> tmp0, tmp1;
  VectorXcd b_vec;
  VectorXd term;

  term = VectorXd::Zero(4);

  for(i = 0; i < M; i++){
      // preparing indexes

      gid1 = vid2gid_st[i][0];
      gid2 = vid2gid_st[i][1];

      // updating for gid1

      tmp0 = (wvis(i)*g(gid1)*conj(h(gid2))
             + wvis(i)*h(gid1)*conj(g(gid2)))/2.0
             - wy(i);
      term(0) += norm(tmp0);
  }

  for(i = 0; i<Gnum; i++){

    // preparing indexes

    st = gid2vid_tbl[i][0];
    k  = gid2vid_tbl[i][3];

    if(k >= 0){

      d_t = time_tbl[gid2vid_tbl[k][1]] - time_tbl[gid2vid_tbl[i][1]];

      w_at = 1/(sig(st)*d_t);

      tmp0 = h(k)-g(i);
      tmp1 = g(k)-h(i);

      term(1) += w_at*(norm(tmp0) + norm(tmp1));

      tmpd0 = abs(h(k)) - abs(g(i));
      tmpd1 = abs(g(k)) - abs(h(i));

      term(2) += w_at*(tmpd0*tmpd0 + tmpd1*tmpd1);
    }

    tmp0 = (g(i)-h(i));

    term(3) += norm(tmp0);
  }

  cost = (term(0) + lambda_1*term(1) + lambda_2*term(2) + rho*term(3))/2.0;

  if(prt == 1){
  //  cout << "  chi-sq          " << term(0) << endl;
  //  cout << "lambda_1: " << lambda_1 << endl;
  //  cout << "  (gt-ht-1)^2     " << term(1) << endl;
  //  cout << "lambda_2: " << lambda_2 << endl;
  //  cout << "  (|gt|-|ht-1|)^2 " << term(2) << endl;
  //  cout << "rho:      " << rho << endl;
    cout << "  (g-h)^2         " << term(3) << endl;
  }

  return(cost);
}

double cost_g(VectorXcd &wvis, VectorXcd &wy, VectorXd &sig,
              VectorXcd &g,
              int **gid2vid_tbl, int **vid2gid_st, double *time_tbl,
              int M, int Gnum,
              double lambda_1, double lambda_2, int prt)
{
  int      i, gid1, gid2, st, k;
  double   d_t, w_at, cost, tmpd;
  complex<double> tmp;
  VectorXd term;

  term = VectorXd::Zero(3);

  for(i = 0; i < M; i++){
      gid1 = vid2gid_st[i][0];
      gid2 = vid2gid_st[i][1];

      tmp = wvis(i)*g(gid1)*conj(g(gid2)) - wy(i);
      term(0) += norm(tmp);
  }

  term(0) /= 2.0;

  for(i = 0; i<Gnum; i++){
    st = gid2vid_tbl[i][0];
    k  = gid2vid_tbl[i][3];

    if(k >= 0){
      d_t = time_tbl[gid2vid_tbl[k][1]] - time_tbl[gid2vid_tbl[i][1]];

      w_at     = 1/(sig(st)*d_t);

      tmp      = g(k) - g(i);
      term(1) += w_at*norm(tmp);

      tmpd     = abs(g(k)) - abs(g(i));
      term(2) += w_at*tmpd*tmpd;
    }
  }

  cost = term(0) + lambda_1*term(1) + lambda_2*term(2);

  if(prt == 1){
    cout << endl;
    cout << "  wieghted chi-sq " << 2*term(0) << endl;
    cout << "lambda_1: " << lambda_1 << endl;
    cout << "  (gt-gt-1)^2     " << term(1) << endl;
    cout << "lambda_2: " << lambda_2 << endl;
    cout << "  (|gt|-|gt-1|)^2 " << term(2) << endl;
    cout << "total cost        " << cost    << endl;
    cout << endl;
  }

  return(cost);
}


void update_g(VectorXd &pow_wvis, VectorXcd &wyvis_conj, VectorXd &sig,
              VectorXcd &g, VectorXcd &h,
              int **gid2vid_tbl, int **vid2gid_st, double *time_tbl,
              int M, int Gnum, int Stnum,
              double lambda_1, double lambda_2, double rho,
              VectorXcd &gnew)
{
  int i, gid1, gid2, st, k, g_zeros;
  double d_t, w_at;
  VectorXd h_abs, h_pow, a_vec, c_vec, b_abs, z_vec, ainv_vec;
  VectorXcd b_vec;

  a_vec = VectorXd::Zero(Gnum);
  b_vec = VectorXcd::Zero(Gnum);
  c_vec = VectorXd::Zero(Gnum);
  ainv_vec = VectorXd::Zero(Gnum);

//  h_abs = h.cwiseAbs();
  h_pow = h.cwiseAbs2();
  h_abs = h.cwiseAbs();

  for(i = 0; i < M; i++){

      gid1 = vid2gid_st[i][0];
      gid2 = vid2gid_st[i][1];

      // updating for gid1

      a_vec(gid1) += pow_wvis(i)*(h_pow(gid2));
      b_vec(gid1) += wyvis_conj(i)*(h(gid2));

      // updating for gid2

      a_vec(gid2) += pow_wvis(i)*(h_pow(gid1));
      b_vec(gid2) += conj(wyvis_conj(i))*(h(gid1));
  }

  a_vec /= 2.0;
  a_vec.array() += rho;
  b_vec = (b_vec/2.0) + rho*h;

  for(i = 0; i < Gnum; i++){

    st = gid2vid_tbl[i][0];
    k  = gid2vid_tbl[i][3];

    if(k >= 0){

      d_t = time_tbl[gid2vid_tbl[k][1]] - time_tbl[gid2vid_tbl[i][1]];
      w_at = 1/(sig(st)*d_t);

      a_vec(i) += (lambda_1 + lambda_2)*w_at;
      b_vec(i) +=  lambda_1 * h(k)     *w_at;
      c_vec(i) +=  lambda_2 * h_abs(k) *w_at;

      a_vec(k) += (lambda_1 + lambda_2)*w_at;
      b_vec(k) +=  lambda_1 * h(i)     *w_at;
      c_vec(k) +=  lambda_2 * h_abs(i) *w_at;
    }
  }

  ainv_vec.array() = 1/a_vec.array();

  b_abs = b_vec.cwiseAbs();

  z_vec = (c_vec.array() + b_abs.array())*ainv_vec.array();

//  g_zeros = Gnum - adjust_g(Gnum,ainv_vec,z_vec);

  g_zeros = Gnum - adjust_g_st(Gnum, Stnum, gid2vid_tbl, ainv_vec, z_vec);

  if(g_zeros>0){
    cout << g_zeros << " components were set to 0" << endl;
  }

  for(i = 0; i < Gnum; ++i){
    if(b_abs(i) > 0){
      gnew(i) = z_vec(i)*b_vec(i)/b_abs(i);
    }
  }

}

double self_calibration(double *vis_r, double *vis_i, double *vis_std,
                        double *y_r, double *y_i,
                        double *ginit_r, double *ginit_i,
                        int **gid2vid_tbl, int **vid2gid_st,
                        double *time_tbl,
                        int M, int Gnum, int Stnum, int Tnum,
                        double lambda_1, double lambda_2,
                        double rho_init,
                        int maxiter, double eps,
                        double *gain_r, double *gain_i)
{
  int i, j, flag = 0;
  VectorXd a_vec, c_vec, pow_wvis, sig;
  VectorXcd wvis, wy, wyvis_conj, g, h, b_vec, gnew;
  double gh_diff_th = 1.0e-10, cost, cost_new = 0, rho;

  rho = rho_init;

  cout << endl << "Self Calibration" << endl;

  wvis = VectorXcd::Zero(M);
  wy   = VectorXcd::Zero(M);
  g    = VectorXcd::Zero(Gnum);
  h    = VectorXcd::Zero(Gnum);
  gnew = VectorXcd::Zero(Gnum);

  a_vec = VectorXd::Zero(Gnum);
  b_vec = VectorXcd::Zero(Gnum);
  c_vec = VectorXd::Zero(Gnum);

  sig = VectorXd::Ones(Stnum);

  pow_wvis   = VectorXd::Zero(Gnum);
  wyvis_conj = VectorXcd::Zero(Gnum);

  for(i = 0; i < M; i++){
    wvis(i)  = complex<double>(vis_r[i],vis_i[i])/vis_std[i];
    wy(i)    = complex<double>(y_r[i],y_i[i])/vis_std[i];
  }

  for(i = 0; i < Gnum; i++) g(i) = complex<double>(ginit_r[i],ginit_i[i]);

  pow_wvis   = wvis.cwiseAbs2();
  wyvis_conj = wy.array()*(conj(wvis.array()));

  h = g;

  if(maxiter<MINITER) maxiter = MINITER;

  cout << endl;
  cout <<  "   Number of visibility: " << M << endl;
  cout <<  "   Number of stations:   " << Stnum << endl;
  cout <<  "   Number of obs time:   " << Tnum << endl;
  cout <<  "   Number of gains:      " << Gnum << endl;
  cout <<  endl;
  cout <<  "Parameters for self-calibration" << endl;
  cout << endl;
  cout <<  "   lambda_1:             " << lambda_1 << endl;
  cout <<  "   lambda_2:             " << lambda_2 << endl;
  cout <<  "   rho:                  " << rho << endl;
  cout << endl;
  cout <<  "   maxiter:              " << maxiter << endl;
  cout <<  "   epsilon:              " << eps << endl;
  cout << endl;

  cost = cost_g(wvis, wy, sig, g, gid2vid_tbl, vid2gid_st, time_tbl,
               M, Gnum, lambda_1, lambda_2, 1);

  for(j = 0; j < 30; j++){
    cout << " rho is: " << rho << endl << endl;
    for(i = 0; i < maxiter; ++i){

      update_g(pow_wvis, wyvis_conj, sig, g, h,
               gid2vid_tbl, vid2gid_st, time_tbl,
               M, Gnum, Stnum, lambda_1, lambda_2, rho, gnew);

      g = gnew;

      update_g(pow_wvis, wyvis_conj, sig, h, g,
               gid2vid_tbl, vid2gid_st, time_tbl,
               M, Gnum, Stnum, lambda_1, lambda_2, rho, gnew);

      h = gnew;

      cost_new = cost_gh(wvis, wy, sig, g, h,
                    gid2vid_tbl, vid2gid_st, time_tbl,
                    M, Gnum, lambda_1, lambda_2, rho,0);

      if((i % 100) == 0){
        printf("Iteration %d: cost is %f\n",i+1, cost_new);
//        cout << "Iteration " << i+1 << ": cost is " << cost_new << endl;
      }

      if( (i > MINITER) && fabs(cost - cost_new)<eps){
        flag = 1;
        break;
      }
      cost = cost_new;
    }

    if(flag == 1) if((g-h).squaredNorm()/((double)Gnum) < gh_diff_th){
      cost = cost_gh(wvis, wy, sig, g, h,
                    gid2vid_tbl, vid2gid_st, time_tbl,
                    M, Gnum, lambda_1, lambda_2, rho,1);
      break;
    }

    rho  = RHOSTEP*rho;
    cost = cost_new;
  }

  if(flag == 0){
    cout << "not converged" << endl;
  }

  g = (g+h)/2;

  cost = cost_g(wvis, wy, sig, g, gid2vid_tbl, vid2gid_st, time_tbl,
                M, Gnum, lambda_1, lambda_2, 1);

//  cout << "cost is " << cost << endl;

  for(i = 0; i < Gnum; i++){
    gain_r[i] = g(i).real();
    gain_i[i] = g(i).imag();
  }
  return(cost);
}

void modify_visibility(double *vis_r, double *vis_i,
                       double *gain_r, double *gain_i,
                       int M, int Gnum, int **vid2gid_st,
                       double *newvis_r, double *newvis_i)
{
  int i, gid1, gid2;
  complex<double> tmpv;
  VectorXcd g, newvis;

  g      = VectorXcd::Zero(Gnum);
  newvis = VectorXcd::Zero(M);

  cout << "Adjust visibility with estimated gains." << endl;

  for(i = 0; i < Gnum; i++) g(i) = complex<double>(gain_r[i],gain_i[i]);

  for(i = 0; i < M; i++){
    gid1 = vid2gid_st[i][0];
    gid2 = vid2gid_st[i][1];

    tmpv = complex<double>(vis_r[i], vis_i[i]);
    newvis(i) = tmpv*g(gid1)*(conj(g(gid2)));
  }

  for(i = 0; i < M; i++){
    newvis_r[i] = newvis(i).real();
    newvis_i[i] = newvis(i).imag();
  }
}

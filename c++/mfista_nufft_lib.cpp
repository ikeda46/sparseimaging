#include "mfista.hpp"
#include <iomanip>
#include <numeric>

#ifdef _OPENMP
  #include <omp.h>
#endif

#define REDUCTIONNUM 4

// mfist_nufft_lib

inline fftw_complex *fftw_cast(const std::complex<double> *p)
{
  return const_cast<fftw_complex*>(reinterpret_cast<const fftw_complex*>(p));
}

int idx_fftw(int m, int Mr)
{
  if(m >= 0 && m < Mr/2) return(m);
  else if(m >= -Mr/2)    return(m+Mr);
  else                   return(2*Mr);
}

void large2small(int Nx, int Ny, VectorXd &out_s, VectorXd &in_l,
  VectorXd &E4)
{
  int Nxh, Nyh;
  double MMinv;

  Nxh = (int)Nx/2;
  Nyh = (int)Ny/2;

  MMinv  = 1/((double)(4*Nx*Ny));

  //openmp
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for(int k = 0; k < Nxh; ++k){

    out_s.segment(k*Ny,Ny) <<
      (E4.segment(k*Ny,    Nyh).array())
        *(in_l.segment(2*(k+3*Nxh)*Ny+3*Nyh,Nyh).array())*MMinv,
      (E4.segment(k*Ny+Nyh,Nyh).array())
        *(in_l.segment(2*(k+3*Nxh)*Ny,      Nyh).array())*MMinv;

    out_s.segment((k+Nxh)*Ny, Ny) <<
      (E4.segment((k+Nxh)*Ny, Nyh).array())
        *(in_l.segment(2*k*Ny+3*Nyh, Nyh).array())*MMinv,
      (E4.segment((k+Nxh)*Ny+Nyh,   Nyh).array())
        *(in_l.segment(2*k*Ny,       Nyh).array())*MMinv;
  }
}

void small2large(int Nx, int Ny, VectorXd &out_l, VectorXd &in_s,
  VectorXd &E4)
{
  int Nxh, Nyh;

  Nxh = (int)Nx/2;
  Nyh = (int)Ny/2;

  out_l.setZero();

  //openmp
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for(int k = 0; k < Nxh; ++k){

    out_l.segment(2*Ny*(k+3*Nxh)+3*Nyh, Nyh) =
      (in_s.segment(k*Ny,           Nyh).array())
      *(E4.segment(k*Ny,           Nyh).array());

    out_l.segment(2*Ny*(k+3*Nxh),       Nyh) =
      (in_s.segment(k*Ny      +Nyh, Nyh).array())
      *(E4.segment(k*Ny      +Nyh, Nyh).array());

    out_l.segment(2*Ny*k        +3*Nyh, Nyh) =
      (in_s.segment((k+Nxh)*Ny,     Nyh).array())
      *(E4.segment((k+Nxh)*Ny,     Nyh).array());

    out_l.segment(2*Ny*k,               Nyh) =
      (in_s.segment((k+Nxh)*Ny+Nyh, Nyh).array())
      *(E4.segment((k+Nxh)*Ny+Nyh, Nyh).array());
  }
}

void preNUFFT(int M, int Nx, int Ny, VectorXd &u, VectorXd &v,
  VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y,
  VectorXd &E4, VectorXi &mx, VectorXi &my, VectorXi &cover_o, VectorXi &cover_c)
{
  int i, j, k;
  double taux, tauy, coeff, xix, xiy, Mrx, Mry, tmp3x, tmp3y,
    tmpx, tmpy, pi, tmpi, tmpj, tmpcoef[MSP2];
  VectorXi idx, idx_c;

  cover_o.setZero();
  cover_c.setZero();

  pi = M_PI;

  Mrx = (double) 2*Nx;
  Mry = (double) 2*Ny;

  taux = 12/((double)(Nx*Nx));
  tauy = 12/((double)(Ny*Ny));

  coeff = pi/sqrt(taux*tauy);

  tmp3x = pow((pi/Mrx),2.0)/taux;
  tmp3y = pow((pi/Mry),2.0)/tauy;

  for(j = 0; j < MSP2; ++j) tmpcoef[j] = (double)(-MSP+1+j);

  #ifdef _OPENMP
  #pragma omp parallel
  {
  #pragma omp for
  #endif
  for(k = 0; k < M; ++k){

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

    for(j = 0; j < MSP2; ++j){
      E2x(j, k) = exp(tmpcoef[j]*tmpx-tmp3x*((double)((j-MSP+1)*(j-MSP+1))));
      E2y(j, k) = exp(tmpcoef[j]*tmpy-tmp3y*((double)((j-MSP+1)*(j-MSP+1))));
    }
  }

  #ifdef _OPENMP
  #pragma omp for
  #endif
  for(k = 0; k < M; ++k){
      if(idx_fftw( (my(k)-MSP+1),Mry) < Ny+1 || idx_fftw( (my(k)+MSP),Mry) < Ny+1) cover_o(k) = 1;
      if(idx_fftw(-(my(k)-MSP+1),Mry) < Ny+1 || idx_fftw(-(my(k)+MSP),Mry) < Ny+1) cover_c(k) = 1;
  }

  #ifdef _OPENMP
  #pragma omp for
  #endif
  for(i = 0; i < Nx; ++i){

    tmpi = (double)(i-Nx/2);
    tmpx = taux*tmpi*tmpi;

    for(j = 0; j< Ny; ++j){
      tmpj = (double)(j-Nx/2);
      tmpy = tauy*tmpj*tmpj;
      E4(i*Ny + j) = coeff*exp(tmpx + tmpy);
    }
  }

  #ifdef _OPENMP
  }
  #endif
}

complex<double> map_0(complex<double> x){return(x);}
complex<double> map_c(complex<double> x){return(conj(x));}

void NUFFT2d1(int M, int Nx, int Ny, VectorXd &Xout,
  VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y, VectorXd &E4,
  VectorXi &mx, VectorXi &my, VectorXi &cover_o, VectorXi &cover_c,
  VectorXcd &in, VectorXd &out, fftw_plan *fftwplan_c2r, VectorXcd &Fin,
  MatrixXcd &mbuf_l, std::vector<int> const &tile_boundary)
{
  int Mh, threadnum;
//  complex<double> v0, vy, tmpc;
//  complex<double> vy, tmpc;
  complex<double> (*map_nufft)(complex<double> x);

  Mh =  Ny + 1;

  mbuf_l.setZero();

  if(NU_SIGN == -1) map_nufft  = map_c;
  else              map_nufft  = map_0;

  if (!tile_boundary.empty()) {
    // openmp should be available
    int const ntile = tile_boundary.size() - 1;
    #pragma omp parallel for schedule(dynamic)
    for (int itile = 0; itile < ntile; ++itile) {
      for (int k = 0; k < M; ++k) {
        int const myk = my(k);
        if (cover_o(k) == 1) {
          int const col_from = myk + MSP + 1 + Ny;
          int const col_to = col_from + MSP2;
          int const tile_from = tile_boundary[itile];
          int const tile_to = tile_boundary[itile + 1];
          if (tile_from < col_to && col_from < tile_to) {
            int const icol = (tile_from > col_from) ? tile_from : col_from;
            int const ncol = ((tile_to < col_to) ? tile_to : col_to) - icol;
            complex<double> v0half = 0.5 * E1(k)*(map_nufft(Fin(k)));
            int const ix = mx(k) + MSP + 1 + Nx;
            if (ncol == MSP2) {
              mbuf_l.block<MSP2, MSP2>(ix, icol) +=
                v0half
                * E2x.block<MSP2, 1>(0, k)
                * E2y.block<MSP2, 1>(0, k).transpose();
            } else {
              for (int jcol = 0; jcol < ncol; ++jcol) {
                mbuf_l.block<MSP2, 1>(ix, icol + jcol) +=
                  v0half
                  * E2x.block<MSP2, 1>(0, k)
                  * E2y(icol - col_from + jcol, k);
              }
            }
          }
        }
        if (cover_c(k) == 1) {
          int const col_from = - myk + MSP + Ny;
          int const col_to = col_from + MSP2;
          int const tile_from = tile_boundary[itile];
          int const tile_to = tile_boundary[itile + 1];
          if (tile_from < col_to && col_from < tile_to) {
            int const icol = (tile_from > col_from) ? tile_from : col_from;
            int const ncol = ((tile_to < col_to) ? tile_to : col_to) - icol;
            complex<double> v0half = 0.5 * conj(E1(k)*(map_nufft(Fin(k))));
            int const ix = - mx(k) + MSP + Nx;
            if (ncol == MSP2) {
              mbuf_l.block<MSP2, MSP2>(ix, icol) +=
                v0half
                * E2x.block<MSP2, 1>(0, k).colwise().reverse()
                * E2y.block<MSP2, 1>(0, k).colwise().reverse().transpose();
            } else {
              int const offset = max(0, tile_from - col_from);
              for (int jcol = 0; jcol < ncol; ++jcol) {
                mbuf_l.block<MSP2, 1>(ix, icol + jcol) +=
                  v0half
                  * E2x.block<MSP2, 1>(0, k).colwise().reverse()
                  * E2y(MSP2 - 1 - offset - jcol, k);
              }
            }
          }
        }
      }
    }
  } else {
    // openmp
    #ifdef _OPENMP

      threadnum = omp_get_max_threads();
      if(threadnum > REDUCTIONNUM) threadnum = REDUCTIONNUM;

      #pragma omp declare reduction(+:Eigen::MatrixXcd:omp_out+=omp_in) initializer(omp_priv = omp_orig)
      #pragma omp parallel for num_threads(threadnum) reduction(+:mbuf_l)
    #endif
    for(int k = 0; k < M; ++k){
      complex<double> v0 = E1(k)*(map_nufft(Fin(k)));

      if(cover_o(k)==1)
        mbuf_l.block<MSP2,MSP2>( mx(k)+MSP+1+Nx, my(k)+MSP+1+Ny) +=
          0.5*v0
          *E2x.block<MSP2,1>(0,k)
          *E2y.block<MSP2,1>(0,k).transpose();

      if(cover_c(k)==1)
        mbuf_l.block<MSP2,MSP2>(-mx(k)+MSP+Nx,-my(k)+MSP+Ny) +=
          0.5*(conj(v0))
          *E2x.block<MSP2,1>(0,k).colwise().reverse()
          *E2y.block<MSP2,1>(0,k).colwise().reverse().transpose();
    }
  }

  #ifdef _OPENMP
  #pragma omp parallel for
  for (size_t i = 0; i < mbuf_l.rows(); ++i) {
      mbuf_l(i, MSP2 + 2 * Ny) = mbuf_l(i, MSP2);
  }
  #else
  mbuf_l.block(0,MSP2+2*Ny,2*Nx+MSP4,1) = mbuf_l.block(0,MSP2,2*Nx+MSP4,1);
  #endif

  //openmp
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for(int i = 0; i < Nx; ++i){
    in.segment(i*Mh     ,Mh) =
      mbuf_l.block(i+MSP2+Nx,MSP2+Ny,1,Mh).transpose();
    in.segment((i+Nx)*Mh,Mh) =
      mbuf_l.block(i+MSP2,   MSP2+Ny,1,Mh).transpose();
  }

  fftw_execute(*fftwplan_c2r);

  large2small(Nx, Ny, Xout, out, E4);
}

void NUFFT2d2(int M, int Nx, int Ny, VectorXcd &Fout, VectorXd &E1,
  MatrixXd &E2x, MatrixXd &E2y, VectorXd &E4,
  VectorXi &mx, VectorXi &my,
 	VectorXd &in, VectorXcd &out, fftw_plan *fftwplan_r2c, VectorXd &Xin, MatrixXcd &mbuf_h)
{
  int Mrx, Mry, Mh;
  double MMinv;
  complex<double> (*map0_nufft)(complex<double> x),(*mapc_nufft)(complex<double> x);

  if(NU_SIGN == -1) {map0_nufft  = map_c; mapc_nufft = map_0;}
  else              {map0_nufft  = map_0; mapc_nufft = map_c;}

  Mrx = 2*Nx;
  Mry = 2*Ny;
  Mh  = Ny+1;
  MMinv = 1/((double)Mrx*Mry);

  small2large(Nx, Ny, in, Xin, E4);

  fftw_execute(*fftwplan_r2c);

  //openmp
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for(int i = 0; i < Nx; ++i){
    mbuf_h.block(0,i+Nx,Mh,1) = out.segment(i*Mh,     Mh);
    mbuf_h.block(0,i,   Mh,1) = out.segment((i+Nx)*Mh,Mh);
  }

  //openmp
  #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
  #endif
  for(int k = 0; k < M; ++k){

    double st0 = idx_fftw( (my(k)-MSP+1),Mry);
    double ed0 = idx_fftw( (my(k)+MSP),  Mry);

    double stc = idx_fftw(-(my(k)+MSP),  Mry);
    double edc = idx_fftw(-(my(k)-MSP+1),Mry);

    if((st0 < ed0) && st0 >=0 && ed0 <= Mh ){

      Fout(k) = MMinv*E1(k)*map0_nufft(
        ((E2y.block<MSP2,1>(0,k)*(E2x.block<MSP2,1>(0,k).transpose())).array()
          *(mbuf_h.block<MSP2,MSP2>(st0,mx(k)-MSP+1+Nx)).array()).sum()
        );
    }
    else if((stc < edc) && stc >=0 && edc <= Mh ){

      Fout(k) = MMinv*E1(k)*mapc_nufft(
        (((E2y.block<MSP2,1>(0,k).colwise().reverse())
            *(E2x.block<MSP2,1>(0,k).colwise().reverse().transpose())).array()
          *(mbuf_h.block<MSP2,MSP2>(stc,-mx(k)-MSP+Nx)).array()).sum()
        );

    }
    else{
      Fout(k) = 0;
      for(int ly = 0; ly < MSP2; ++ly){

        int j = idx_fftw((my(k)+ly-MSP+1),Mry);
        double tmp = MMinv*E1(k)*E2y(ly,k);

        if(j < (Ny+1)){
          Fout(k) +=
            map0_nufft(
              tmp
              *(E2x.block<MSP2,1>(0,k).transpose().dot(
                (mbuf_h.block<1,MSP2>(j,mx(k)-MSP+1+Nx).transpose())))
              );
        }
        else{
          j = idx_fftw(-(my(k)+ly-MSP+1),Mry);
          if(j < (Ny+1)){
            Fout(k) +=
              mapc_nufft(
                tmp
                *(E2x.block<MSP2,1>(0,k).transpose().dot(
                  (mbuf_h.block<1,MSP2>(j,-mx(k)-MSP+Nx).rowwise().reverse().transpose())))
                );
          }
        }
      }
    }
  }
}

double calc_F_part_nufft(int M, int Nx, int Ny,
  VectorXcd &yAx,
  VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y, VectorXd &E4,
  VectorXi &mx, VectorXi &my,
  VectorXd &in_r, VectorXcd &out_c,
  fftw_plan *fftwplan_r2c, VectorXcd &vis, VectorXd &weight, VectorXd &xvec,
  MatrixXcd &mbuf_h)
{
  NUFFT2d2(M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my,
    in_r, out_c, fftwplan_r2c, xvec, mbuf_h);

  #ifdef _OPENMP
  double sqnorm = 0;
  #pragma omp parallel for reduction(+:sqnorm)
  for (size_t i = 0; i < vis.size(); ++i) {
      auto const tmp = (vis[i] - yAx[i]) * weight[i];
      yAx[i] = tmp;
      sqnorm += (tmp.real() * tmp.real() + tmp.imag() * tmp.imag()) / 2;
  }
  return sqnorm;
  #else
  yAx = (vis.array() - yAx.array())*weight.array();

  return(yAx.squaredNorm()/2);
  #endif
}

void dF_dx_nufft(int M, int Nx, int Ny, VectorXd &dFdx,
		 VectorXd &E1, MatrixXd &E2x, MatrixXd &E2y,
		 VectorXd &E4,
		 VectorXi &mx, VectorXi &my, VectorXi &cover_o, VectorXi &cover_c,
		 VectorXcd &in_c, VectorXd &out_r, fftw_plan *fftwplan_c2r,
		 VectorXd &weight, VectorXcd &yAx, MatrixXcd &mbuf_l,
         std::vector<int> const &tile_boundary)
{
  yAx.array() *= weight.array();

  NUFFT2d1(M, Nx, Ny, dFdx, E1, E2x, E2y, E4, mx, my, cover_o, cover_c,
	   in_c, out_r, fftwplan_c2r, yAx, mbuf_l, tile_boundary);
}

/* TSV */

void sort_input(int const M, int const Nx, int const Ny, double *u_dx, double *v_dy,
                double *vis_r, double *vis_i, double *vis_std)
{
  int const Mrx = 2 * Nx;
  int const Mry = 2 * Ny;
  std::vector<size_t> index_array(M);
  std::iota(index_array.begin(), index_array.end(), 0);
  std::sort(index_array.begin(), index_array.end(),
    [&](size_t a, size_t b) {
      auto const v_a = round(v_dy[a] * Mry / (2 * M_PI));
      auto const v_b = round(v_dy[b] * Mry / (2 * M_PI));
      if (v_a == v_b) {
        auto const u_a = round(u_dx[a] * Mrx / (2 * M_PI));
        auto const u_b = round(u_dx[b] * Mrx / (2 * M_PI));
        return u_a < u_b;
      }
      else {
        return v_a < v_b;
      }
    }
  );

  #ifdef _OPENMP
  #pragma omp parallel sections
  #endif
  {
    #ifdef _OPENMP
    #pragma omp section
    #endif
    {
      std::vector<double> tmp(M);
      for (size_t i = 0; i < M; ++i) {
          tmp[i] = u_dx[i];
      }
      for (size_t i = 0; i < M; ++i) {
          u_dx[i] = tmp[index_array[i]];
      }
    }

    #ifdef _OPENMP
    #pragma omp section
    #endif
    {
      std::vector<double> tmp(M);
      for (size_t i = 0; i < M; ++i) {
          tmp[i] = v_dy[i];
      }
      for (size_t i = 0; i < M; ++i) {
          v_dy[i] = tmp[index_array[i]];
      }
    }

    #ifdef _OPENMP
    #pragma omp section
    #endif
    {
      std::vector<double> tmp(M);
      for (size_t i = 0; i < M; ++i) {
          tmp[i] = vis_r[i];
      }
      for (size_t i = 0; i < M; ++i) {
          vis_r[i] = tmp[index_array[i]];
      }
    }

    #ifdef _OPENMP
    #pragma omp section
    #endif
    {
      std::vector<double> tmp(M);
      for (size_t i = 0; i < M; ++i) {
          tmp[i] = vis_i[i];
      }
      for (size_t i = 0; i < M; ++i) {
          vis_i[i] = tmp[index_array[i]];
      }
    }

    #ifdef _OPENMP
    #pragma omp section
    #endif
    {
      std::vector<double> tmp(M);
      for (size_t i = 0; i < M; ++i) {
          tmp[i] = vis_std[i];
      }
      for (size_t i = 0; i < M; ++i) {
          vis_std[i] = tmp[index_array[i]];
      }
    }
  }
}


void configure_tile(int npixels, int nthreads, std::vector<int> &tile_boundary)
{
    // innermost tile size and number of tiles
    int tile_size = MSP; // MSP = MSP2 / 2;
    int const nadd_center = nthreads / 3;

    // 4-thread reduction seems to be efficient for less number of cores
    if (nthreads < 12 || npixels < tile_size * nadd_center * 2 * 2) {
        return;
    }

    int const middle = npixels / 2 + npixels % 2;
    int left = middle;
    int right = middle;
    int np = npixels;

    // innermost tiles
    tile_boundary.push_back(middle);
    for (int i = 0; i < nadd_center; ++i) {
        left -= tile_size;
        tile_boundary.insert(tile_boundary.begin(), left);
        right += tile_size;
        tile_boundary.push_back(right);
        np -= 2 * tile_size;
    }

    // gradually increase tile size
    int const nadd = 2;
    while (left > 0) {
        tile_size *= 2;
        for (int i = 0; i < nadd; ++i) {
            left -= tile_size;
            right += tile_size;
            if (left - tile_size <= 0) {
                left = 0;
                right = npixels;
                break;
            } else {
                tile_boundary.insert(tile_boundary.begin(), left);
                tile_boundary.push_back(right);
            }
        }
    }

    tile_boundary.insert(tile_boundary.begin(), 0);
    tile_boundary.push_back(npixels);

    // std::cout << "npixels = " << npixels << " nthreads = " << nthreads << std::endl;
    // std::cout << "tile_boundary: " << tile_boundary.size() << " boundaries" << std::endl;
    // char delim = '[';
    // for (auto i = tile_boundary.begin(); i != tile_boundary.end(); ++i) {
    //     std::cout << delim << " " << *i;
    //     delim = ',';
    // }
    // std::cout << " ]" << std::endl;
    // std::cout << "   ";
    // for (size_t i = 0; i < tile_boundary.size() - 1; ++i) {
    //     std::cout << tile_boundary[i + 1] - tile_boundary[i] << "   ";
    // }
    // std::cout << std::endl;
}


void configure_tile(
    int npixels, int nthreads, int Ny,
    VectorXi const &my, VectorXi const &cover_o, VectorXi const &cover_c,
    std::vector<int> &tile_boundary)
{
    // make histogram
    std::vector<unsigned long> hist(npixels, 0ul);
    for (size_t k = 0; k < my.size(); ++k) {
        int const myk = my(k);
        if (cover_o(k)) {
            int const col_from = myk + MSP + 1 + Ny;
            int const col_to = col_from + MSP2;
            for (int i = col_from; i < col_to; ++i) {
                hist[i] += 1;
            }
        }
        if (cover_c(k)) {
            int const col_from = - myk + MSP + Ny;
            int const col_to = col_from + MSP2;
            for (int i = col_from; i < col_to; ++i) {
                hist[i] += 1;
            }
        }
    }
    unsigned long const hist_total = std::accumulate(hist.begin(), hist.end(), 0ul);

    // number of data per tile
    unsigned long const ndata = hist_total / nthreads + ((hist_total % nthreads > 0) ? 1 : 0);

    // configure tile
    tile_boundary.clear();
    tile_boundary.push_back(0);
    unsigned long count = 0;
    for (int i = 0; i < npixels && tile_boundary.size() < nthreads; ++i) {
        if (count > ndata) {
            tile_boundary.push_back(i);
            count = 0;
        }
        count += hist[i];
    }
    tile_boundary.push_back(npixels);

    // std::cout << "hist: total " << hist_total << " ndata " << ndata << std::endl;
    // char delim2 = '[';
    // for (auto i = hist.begin(); i != hist.end(); ++i) {
    //     std::cout << delim2 << " " << *i;
    //     delim2 = ',';
    // }
    // std::cout << " ]" << std::endl;

    // std::cout << "npixels = " << npixels << " nthreads = " << nthreads << std::endl;
    // std::cout << "tile_boundary: " << tile_boundary.size() << " boundaries" << std::endl;
    // char delim = '[';
    // for (auto i = tile_boundary.begin(); i != tile_boundary.end(); ++i) {
    //     std::cout << delim << " " << *i;
    //     delim = ',';
    // }
    // std::cout << " ]" << std::endl;
    // std::cout << "   ";
    // for (size_t i = 0; i < tile_boundary.size() - 1; ++i) {
    //     std::cout << tile_boundary[i + 1] - tile_boundary[i] << "   ";
    // }
    // std::cout << std::endl;
}


int mfista_L1_TSV_core_nufft(double *xout,
			     int M, int Nx, int Ny, double *u_dx, double *v_dy,
			     int maxiter, double eps,
			     double *vis_r, double *vis_i, double *vis_std,
			     double lambda_l1, double lambda_tsv,
			     double *cinit, double *xinit,
			     int nonneg_flag, int box_flag, float *cl_box)
{
  void (*soft_th_box)(VectorXd &newvec, VectorXd &vector, double eta, int box_flag, VectorXd &box);

  int NN = Nx*Ny, i, iter;
  double Qcore, Fval, Qval, c, tmpa, tmpb, l1cost, tsvcost, costtmp,
    mu=1, munew, *rvecf;
  fftw_complex *cvecf;
  fftw_plan fftwplan_c2r, fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  VectorXi mx, my, cover_o, cover_c;
  VectorXd E1, E4, cost, rvec, xtmp, xnew, zvec, dfdx, dtmp, weight, xvec,
           box, u, v, buf_diff;
  VectorXcd yAx, vis, cvec;
  MatrixXd E2x, E2y;
  MatrixXcd mbuf_l, mbuf_h;

  sort_input(M, Nx, Ny, u_dx, v_dy, vis_r, vis_i, vis_std);

  cout << "Memory allocation and preparations." << endl << endl;

  mx      = VectorXi::Zero(M);
  my      = VectorXi::Zero(M);
  cover_c = VectorXi::Zero(M);
  cover_o = VectorXi::Zero(M);
  E1      = VectorXd::Zero(M);
  E2x     = MatrixXd::Zero(MSP2, M);
  E2y     = MatrixXd::Zero(MSP2, M);
  E4      = VectorXd::Zero(NN);

  rvec    = VectorXd::Zero(4*NN);
  cost    = VectorXd::Zero(maxiter);
  dfdx    = VectorXd::Zero(NN);
  xnew    = VectorXd::Zero(NN);
  xtmp    = VectorXd::Zero(NN);
  dtmp    = VectorXd::Zero(NN);
  box     = VectorXd::Zero(NN);

  yAx     = VectorXcd::Zero(M);
  vis     = VectorXcd::Zero(M);
  cvec    = VectorXcd::Zero(2*Nx*(Ny+1));
  mbuf_l  = MatrixXcd::Zero(2*Nx+MSP4,2*Ny+MSP4);
  mbuf_h  = MatrixXcd::Zero(Ny+1,2*Nx);

  weight  = VectorXd::Zero(M);

  buf_diff = VectorXd::Zero(Nx-1);

  u = Map<VectorXd>(u_dx,M);
  v = Map<VectorXd>(v_dy,M);
  xvec = Map<VectorXd>(xinit,NN);
  zvec = xvec;

  if(box_flag == 1){
    for(i = 0; i < NN; ++i) box(i) = (double)cl_box[i];
  }
  else box = VectorXd::Zero(NN);

  for(i = 0; i < M; ++i){
    vis(i) = complex<double>(vis_r[i],vis_i[i]);
    weight(i) = 1/vis_std[i];
  }

  //openmp
  std::vector<int> tile_boundary;
  #ifdef _OPENMP
    cout << "(OPENMP): Running in multi-thread with " <<
    omp_get_max_threads() << " threads." << endl << endl;
    // configure_tile(mbuf_l.cols(), omp_get_max_threads(), tile_boundary);
  #endif

  cout << "Preparation for FFT.";

  // prepare for nufft

  preNUFFT(M, Nx, Ny, u, v, E1, E2x, E2y, E4, mx, my, cover_o, cover_c);

  #ifdef _OPENMP
    configure_tile(mbuf_l.cols(), omp_get_max_threads(), Ny, my, cover_o, cover_c, tile_boundary);
  #endif

  // for fftw

  rvecf = &rvec[0];
  cvecf = fftw_cast(cvec.data());

  //openmp
  #ifdef _OPENMP
  if(fftw_init_threads()){
    fftw_plan_with_nthreads(omp_get_max_threads());
  }
  #endif

  fftwplan_c2r = fftw_plan_dft_c2r_2d(2*Nx,2*Ny, cvecf, rvecf, fftw_plan_flag);
  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvecf, cvecf, fftw_plan_flag);

  fftw_execute(fftwplan_r2c);
  fftw_execute(fftwplan_c2r);

  // initialization

  if(nonneg_flag == 0)
    soft_th_box =soft_threshold_box;
  else if(nonneg_flag == 1)
    soft_th_box =soft_threshold_nonneg_box;
  else {
    cout << "nonneg_flag must be chosen properly." << endl;
    return(0);
  }

  cout << " Done." << endl << endl;

  c = *cinit;

  cout << "Computing image with MFISTA using NUFFT." << endl;
  cout << "Stop if iter = " << maxiter << " or Delta_cost < " << eps
  << endl << endl;

  // main

  costtmp = calc_F_part_nufft(M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my,
			      rvec, cvec, &fftwplan_r2c, vis, weight, xvec, mbuf_h);

  l1cost = xvec.lpNorm<1>();
  costtmp += lambda_l1*l1cost;

  if( lambda_tsv > 0 ){
    tsvcost = TSV(Nx, Ny, xvec, buf_diff);
    costtmp += lambda_tsv*tsvcost;
  }

  for(iter = 0; iter < maxiter; iter++){

    cost(iter) = costtmp;

    if((iter % 10) == 0){
      cout << std::setw(5) << iter+1 << " cost = " << fixed << setprecision(5)
      << cost(iter) << ", c = " << c << endl;
    }

    Qcore = calc_F_part_nufft(M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my,
      rvec, cvec, &fftwplan_r2c, vis, weight, zvec, mbuf_h);

    dF_dx_nufft(M, Nx, Ny, dfdx, E1, E2x, E2y, E4, mx, my, cover_o, cover_c,
      cvec, rvec, &fftwplan_c2r, weight, yAx, mbuf_l, tile_boundary);

    if( lambda_tsv > 0.0 ){
      tsvcost = TSV(Nx, Ny, zvec, buf_diff);
      Qcore += lambda_tsv*tsvcost;

      d_TSV(dtmp, Nx, Ny, zvec);
      dfdx -= lambda_tsv*dtmp;
    }

    for(i = 0; i < maxiter; ++i){
      xtmp = zvec + dfdx/c;
      soft_th_box(xnew, xtmp, lambda_l1/c, box_flag, box);

      Fval = calc_F_part_nufft(M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my,
        rvec, cvec, &fftwplan_r2c, vis, weight, xnew, mbuf_h);

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

      zvec = tmpa * xnew + tmpb * zvec;

      xvec = xnew;
    }
    else{

      tmpa = mu/munew;
      tmpb = 1-(mu/munew);

      zvec = tmpa * xnew + tmpb * zvec;

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

  printf("\n");

  *cinit = c;

  for(i = 0; i < NN; ++i) xout[i] = xvec(i);

  cout << "Cleaning fftw plan." << endl;

  fftw_destroy_plan(fftwplan_c2r);
  fftw_destroy_plan(fftwplan_r2c);

  #ifdef _OPENMP
    fftw_cleanup_threads();
  #endif

  cout << resetiosflags(ios_base::floatfield);

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
  double tmp, *rvecf;
  fftw_complex *cvecf;
  fftw_plan fftwplan_r2c;
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

  VectorXi mx, my, cover_o, cover_c;
  VectorXd E1, E4, weight, rvec, x, u, v, buf_diff;
  VectorXcd yAx, vis, cvec;
  MatrixXd E2x, E2y;
  MatrixXcd mbuf_h;

  cout << "Memory allocation and preparations." << endl;

  mx      = VectorXi::Zero(M);
  my      = VectorXi::Zero(M);
  cover_o = VectorXi::Zero(M);
  cover_c = VectorXi::Zero(M);
  E1      = VectorXd::Zero(M);
  E2x     = MatrixXd::Zero(MSP2,M);
  E2y     = MatrixXd::Zero(MSP2,M);
  E4      = VectorXd::Zero(NN);

  mbuf_h  = MatrixXcd::Zero(Ny+1,2*Nx);

  buf_diff = VectorXd::Zero(Nx-1);
  rvec = VectorXd::Zero(4*NN);
  cvec = VectorXcd::Zero(2*Nx*(Ny+1));
  u = Map<VectorXd>(u_dx,M);
  v = Map<VectorXd>(v_dy,M);
  x = Map<VectorXd>(xvec,NN);

  // prepare for nufft

  preNUFFT(M, Nx, Ny, u, v, E1, E2x, E2y, E4, mx, my, cover_o, cover_c);

  // for fftw

  rvecf = &rvec[0];
  cvecf = fftw_cast(cvec.data());

  //openmp
  #ifdef _OPENMP
  if(fftw_init_threads()){
    fftw_plan_with_nthreads(omp_get_max_threads());
  }
  #endif

  fftwplan_r2c = fftw_plan_dft_r2c_2d(2*Nx,2*Ny, rvecf, cvecf, fftw_plan_flag);

  yAx    = VectorXcd::Zero(M);
  vis    = VectorXcd::Zero(M);
  weight = VectorXd::Zero(M);

  for(i = 0; i < M; ++i){
    vis(i) = complex<double>(vis_r[i],vis_i[i]);
    weight(i) = 1/vis_std[i];
  }

  // computing results

  tmp = calc_F_part_nufft(M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my,
 			  rvec, cvec, &fftwplan_r2c, vis, weight, x, mbuf_h);

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

  for(i = 0; i < NN; ++i){
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

  fftw_destroy_plan(fftwplan_r2c);

  #ifdef _OPENMP
    fftw_cleanup_threads();
  #endif
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
  for(i = 0; i < M; ++i) epsilon += vis_r[i]*vis_r[i] + vis_i[i]*vis_i[i];

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

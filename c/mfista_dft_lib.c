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

void mfista_imaging_core(double *y, double *A, 
			 int *M, int *N, int NX, int NY,
			 double lambda_l1, double lambda_tv, double lambda_tsv,
			 double cinit, double *xinit, double *xout,
			 int nonneg_flag, int looe_flag,
			 struct RESULT *mfista_result)
{
  double s_t, e_t;
  int    iter = 0, inc = 1;
  struct timespec time_spec1, time_spec2;

  dcopy_(N, xinit, &inc, xout, &inc);
  
  get_current_time(&time_spec1);

  /* main loop */

  if( lambda_tv == 0){
    iter = mfista_L1_TSV_core(y, A, M, N, NX, NY,
			      lambda_l1, lambda_tsv, cinit, xout, nonneg_flag);
  }
  else if( lambda_tv != 0  && lambda_tsv == 0 ){
    iter = mfista_L1_TV_core(y, A, M, N, NX, NY,
			     lambda_l1, lambda_tv, cinit, xout, nonneg_flag);
  }
  else{
    printf("You cannot set both of lambda_TV and lambda_TSV positive.\n");
    return;
  }
    
  get_current_time(&time_spec2);

  /* main loop end */

  s_t = (double)time_spec1.tv_sec + (10e-10)*(double)time_spec1.tv_nsec;
  e_t = (double)time_spec2.tv_sec + (10e-10)*(double)time_spec2.tv_nsec;

  mfista_result->comp_time = e_t-s_t;
  mfista_result->ITER = iter;
  mfista_result->nonneg = nonneg_flag;

  calc_result(y, A, M, N, NX, NY,
	      lambda_l1, lambda_tv, lambda_tsv, xout, nonneg_flag, looe_flag,
	      mfista_result);

  return;
}

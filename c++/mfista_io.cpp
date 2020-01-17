#include "mfista.hpp"

// I-O part

void init_result(struct IO_FNAMES *mfista_io,
		 struct RESULT *mfista_result)
{
  mfista_io->fft       = 1;
  mfista_io->fft_fname = NULL;
  mfista_io->v_fname   = NULL;
  mfista_io->A_fname   = NULL;
  mfista_io->in_fname  = NULL;
  mfista_io->out_fname = NULL;
  
  mfista_result->M             = 0;
  mfista_result->N             = 0;
  mfista_result->NX            = 0;
  mfista_result->NY            = 0;
  mfista_result->N_active      = 0;
  mfista_result->maxiter       = 0;
  mfista_result->ITER          = 0;
  mfista_result->nonneg        = 0;
  mfista_result->lambda_l1     = 0;
  mfista_result->lambda_tv     = 0;
  mfista_result->lambda_tsv    = 0;
  mfista_result->sq_error      = 0;
  mfista_result->mean_sq_error = 0;
  mfista_result->l1cost        = 0;
  mfista_result->tvcost        = 0;
  mfista_result->tsvcost       = 0;
  mfista_result->finalcost     = 0;
  mfista_result->comp_time     = 0;
  mfista_result->Lip_const     = 0;
}

void cout_result(char *fname,
		 struct IO_FNAMES *mfista_io,
		 struct RESULT *mfista_result)
{
  cout << endl;
  cout << " FFTW file:              " << mfista_io->fft_fname << endl;  

  if(mfista_io->in_fname != NULL)
    cout << " x was initialized with: " << mfista_io->in_fname << endl;

  if(mfista_io->out_fname != NULL)
    cout << " x is saved to:          " << mfista_io->out_fname << endl;

  cout << endl;
  cout << "Output of " << fname << "\n\n";
  
  cout << " size of input vector:   " << mfista_result->M << endl;
  cout << " size of output vector:  " << mfista_result->N << endl;
  if(mfista_result->NX != 0)
    cout << " size of image:          "
	 << mfista_result->NX << " x " << mfista_result->NY << endl;
  
  cout << endl;
  cout << " Settings:\n\n";

  if(mfista_result->nonneg == 1)
    cout << " x is nonnegative.\n\n";

  if(mfista_result->lambda_l1 != 0)
    cout << " Lambda_1:               " << mfista_result->lambda_l1 << endl;

  if(mfista_result->lambda_tsv != 0)
    cout << " Lambda_TSV:             " << mfista_result->lambda_tsv << endl;

  if(mfista_result->lambda_tv != 0)
    cout << " Lambda_TV:              " << mfista_result->lambda_tv << endl;

  cout << " MAXITER:                " << mfista_result->maxiter << endl;

  cout << endl;
  cout << " Results:\n\n";

  cout << " # of iterations:        " << mfista_result->ITER << endl;
  cout << " cost:                   " << mfista_result->finalcost << endl;
  cout << " computaion time[sec]:   " << mfista_result->comp_time << endl;
  cout << " Est. Lipschitzs const:  " << mfista_result->Lip_const << "\n\n";

  cout << " # of nonzero pixels:    " << mfista_result->N_active << endl;
  cout << " Squared Error (SE):     " << mfista_result->sq_error << endl;
  cout << " Mean SE:                " << mfista_result->mean_sq_error << endl;

  if(mfista_result->lambda_l1 != 0)
    cout << " L1 cost:                " << mfista_result->l1cost << endl;

  if(mfista_result->lambda_tsv != 0)
    cout << " TSV cost:               " << mfista_result->tsvcost << endl;

  if(mfista_result->lambda_tv != 0)
    cout << " TV cost:                " << mfista_result->tvcost << endl;

  cout << endl;

}

void write_result(ostream *ofs, char *fname, struct IO_FNAMES *mfista_io,
		  struct RESULT *mfista_result)
{
  *ofs << endl;
  *ofs << " FFTW file:              " << mfista_io->fft_fname << endl;  

  if(mfista_io->in_fname != NULL)
    *ofs << " x was initialized with: " << mfista_io->in_fname << endl;

  if(mfista_io->out_fname != NULL)
    *ofs << " x is saved to:          " << mfista_io->out_fname << endl;

  *ofs << endl;
  *ofs << "Output of " << fname << "\n\n\n";
  
  *ofs << " size of input vector:   " << mfista_result->M << endl;
  *ofs << " size of output vector:  " << mfista_result->N << endl;
  if(mfista_result->NX != 0)
    *ofs << " size of image:          "
	 << mfista_result->NX << " x " << mfista_result->NY << endl;
  
  *ofs << endl;
  *ofs << " Settings:\n\n";

  if(mfista_result->nonneg == 1)
    *ofs << " x is nonnegative.\n\n";

  if(mfista_result->lambda_l1 != 0)
    *ofs << " Lambda_1:               " << mfista_result->lambda_l1 << endl;

  if(mfista_result->lambda_tsv != 0)
    *ofs << " Lambda_TSV:             " << mfista_result->lambda_tsv << endl;

  if(mfista_result->lambda_tv != 0)
    *ofs << " Lambda_TV:              " << mfista_result->lambda_tv << endl;

  *ofs << " MAXITER:                " << mfista_result->maxiter << endl;

  *ofs << endl;
  *ofs << " Results:\n\n";

  *ofs << " # of iterations:        " << mfista_result->ITER << endl;
  *ofs << " cost:                   " << mfista_result->finalcost << endl;
  *ofs << " computaion time[sec]:   " << mfista_result->comp_time << endl;
  *ofs << " Est. Lipschitzs const:  " << mfista_result->Lip_const << "\n\n";

  *ofs << " # of nonzero pixels:    " << mfista_result->N_active << endl;
  *ofs << " Squared Error (SE):     " << mfista_result->sq_error << endl;
  *ofs << " Mean SE:                " << mfista_result->mean_sq_error << endl;

  if(mfista_result->lambda_l1 != 0)
    *ofs << " L1 cost:                " << mfista_result->l1cost << endl;

  if(mfista_result->lambda_tsv != 0)
    *ofs << " TSV cost:               " << mfista_result->tsvcost << endl;

  if(mfista_result->lambda_tv != 0)
    *ofs << " TV cost:                " << mfista_result->tvcost << endl;

  *ofs << endl;

}


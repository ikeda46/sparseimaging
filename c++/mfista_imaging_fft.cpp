#include "mfista.hpp"
#include <fstream>

void usage(char *s)
{
  
  cerr << s
       << " <fft_data fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double c> <X outfile>"
       << " {X initfile} {-nonneg} {-cl_box box_fname} {-fftw_measure} {-log log_fname}"
       << "\n\n";
  
  cerr << "  <fft_data fname>:    file name of fft_file." << endl;
  cerr << "  <double lambda_l1>:  lambda_l1.  Positive."  << endl;
  cerr << "  <double lambda_tv>:  lambda_tv.  Positive."  << endl;
  cerr << "  <double lambda_tsv>: lambda_tsv. Positive."  << endl;
  cerr << "  <double c>:          c.          Positive."  << endl;
  cerr << "  <X outfile>:         file name to write X."  << "\n\n";

  cerr << " Options." << "\n\n";
    
  cerr << "  {X initfile}:        file name of initial X."        << endl;
  cerr << "  {-nonneg}:           Use this if x is nonnegative."  << endl;
  cerr << "  {-maxiter N}:        maximum number of iterations."  << endl;
  cerr << "  {-eps epsilon}:      epsilon to check convergence."  << endl;
  cerr << "  {-cl_box box_fname}: file name of CLEAN box (float)."  << endl;
  cerr << "  {-fftw_measure}:     for FFTW_MEASURE."              << endl;
  cerr << "  {-log log_fname}:    log file name."                 << "\n\n";

  cerr << " This program solves the following problem with FFT" << "\n\n";

  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1" << "\n\n";
  
  cerr << "    or"   << "\n\n";

  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tv TV(x)"
       << "\n\n";
  
  cerr << "    or"   << "\n\n";
  
  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tsv TSV(x)"
       << "\n\n";

  cerr << " and write x to <X out file>" << "\n\n";

  cerr << " If {-nonneg} option is used, x vector is restricted to be nonnegative." << "\n\n";

  cerr << " c is a parameter used for stepsize. Large c makes the algorithm" << endl;
  cerr << " stable but slow. Around 500000 is fine."                         << "\n\n";

  exit(1);
}

// commandline program

int main(int argc, char *argv[]){

  string buf_str, fftw_fname, log_fname, init_fname, box_fname;
  
  unsigned int fftw_plan_flag = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
 
  int M, NN, NX, NY, dnum, i, *u_dx, *v_dy,
    init_flag = 0, box_flag = 0, log_flag = 0, nonneg_flag = 0, maxiter = MAXITER;

  double *vis_r, *vis_i, *vis_std, *xinit, *xvec, 
    cinit, lambda_l1, lambda_tv, lambda_tsv, eps = EPS;

  float *box;
  
  struct IO_FNAMES mfista_io;
  struct RESULT    mfista_result;

  init_result(&mfista_io, &mfista_result);

  // check the number of variables first.

  if (argc<7) usage(argv[0]);
	
  // read parameters

  lambda_l1 = atof(argv[2]);
  cout << "lambda_l1 = " << lambda_l1 << endl;

  lambda_tv = atof(argv[3]);
  cout << "lambda_tv = " << lambda_tv << endl;

  lambda_tsv = atof(argv[4]);
  cout << "lambda_tsv = " << lambda_tsv << endl;

  cinit = atof(argv[5]);
  cout << "c = " << cinit << endl;

  // read options

  for(i = 7; i < argc ; i++){
    if(strcmp(argv[i],"-log") == 0){
      log_flag = 1;
      i++;
      log_fname = argv[i];
    }
    else if(strcmp(argv[i],"-cl_box") == 0){
      box_flag = 1;
      i++;
      box_fname = argv[i];
    }
    else if(strcmp(argv[i],"-maxiter") == 0){
      i++;
      maxiter = atoi(argv[i]);
    }
    else if(strcmp(argv[i],"-eps") == 0){
      i++;
      eps = atof(argv[i]);
    }
    else if(strcmp(argv[i],"-nonneg") == 0){
      nonneg_flag = 1;
    }
    else if(strcmp(argv[i],"-fftw_measure") == 0){
      fftw_plan_flag = FFTW_MEASURE;
    }
    else{
      init_flag = 1;
      init_fname = argv[i];
    }
  }

  if (nonneg_flag == 1)
    cout << "x is nonnegative." << endl;

  if (log_flag ==1)
    cout << "Log will be saved to "
	 << "\"" << log_fname << "\"." << endl;

  cout << endl;

  /* read fftw_data */

  fftw_fname = argv[1];

  cout << "data file name is \"" << fftw_fname << ".\"\n";

  // open a file
  
  ifstream fft_fs(fftw_fname.data());

  if(fft_fs.fail()){
    cerr << "Cannot open \"" << fftw_fname << ".\"\n";
    exit(0);
  }

  // read data
  
  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(), "M  = %d\n", &M)  != 1) exit(0);

  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(), "NX = %d\n", &NX) != 1) exit(0);

  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(), "NY = %d\n", &NY) != 1) exit(0);

  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(), "\n") !=0)              exit(0);

  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(),
	     "u, v, y_r, y_i, noise_std_dev\n") !=0) exit(0);

  getline(fft_fs, buf_str);
  if (sscanf(buf_str.data(), "\n") !=0)              exit(0);

  // print data size

  NN = NX*NY;

  cout << "number of u-v points:  " << M  << endl;
  cout << "X-dim of image:        " << NX << endl;
  cout << "Y-dim of image:        " << NY << endl;
  cout << "Data size Nx x Ny:     " << NN << endl;

  // allocate vectors

  u_dx = new int [M];
  v_dy = new int [M];
  vis_r    = new double [M];
  vis_i    = new double [M];
  vis_std = new double [M];
  xinit = new double [NN];
  xvec  = new double [NN];
  box = new float [NN];
  
  for(i = 0;i < M; i++){
    getline(fft_fs, buf_str);  
    if(sscanf(buf_str.data(), "%d, %d, %lf, %lf, %lf\n",
	      u_dx+i, v_dy+i, vis_r+i, vis_i+i, vis_std+i)!=5){
      
      cerr << "cannot read data." << endl;
      exit(0);
    }
  }

  fft_fs.close();

  // initialize vector

  if (init_flag == 1){ 
    cout << "Initializing x with " << init_fname.data() << endl;

    ifstream init_fs(init_fname.data(), ios::in | ios::binary);
    dnum = (int)(init_fs.readsome((char*)xinit, sizeof(double)*NN))
      /(sizeof(double));
    init_fs.close();

    if(dnum != NN){
      cerr << "Number of read data is shorter than expected." << endl;
      exit(0);
    }
  }
  else
    for(i = 0; i < NN; i++) xinit[i]=0;

  // cleanbox

  if (box_flag == 1){
    cout << "Restrict x with CLEAN box defined in \"" << box_fname.data()
	 << ".\"\n";

    ifstream box_fs(box_fname.data(), ios::in | ios::binary);
    dnum = (int)(box_fs.readsome((char*)box, sizeof(float)*NN))
      /(sizeof(float));
    box_fs.close();

    if(dnum != NN){
      cerr << "Number of read data is shorter than expected." << endl;
      exit(0);
    }
  }

  // main iteration

  mfista_imaging_core_fft(u_dx, v_dy, vis_r, vis_i, vis_std,
			  M, NX, NY, maxiter, eps, lambda_l1, lambda_tv, lambda_tsv, cinit,
			  xinit, xvec, nonneg_flag, fftw_plan_flag, box_flag, box, &mfista_result);

  // write resulting image to a file

  ofstream out_fs(argv[6], ios::out | ios::binary);
  out_fs.write((char*)xvec, sizeof(double)*NN);
  out_fs.close();

  mfista_io.fft_fname = argv[1];
  mfista_io.out_fname = argv[6];

  if(init_flag == 1) mfista_io.in_fname = (char*)init_fname.data();

  cout_result(argv[0], &mfista_io, &mfista_result);

  if(log_flag == 1){
    ofstream log_fs(log_fname.data(), ios::out);
    write_result(&log_fs, argv[0], &mfista_io, &mfista_result);
    log_fs.close();
  }

  // release memory

  delete u_dx;
  delete v_dy;
  delete vis_r;
  delete vis_i;
  delete vis_std;
  delete xinit;
  delete xvec;
  delete box;

}

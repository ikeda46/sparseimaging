#include "mfista.hpp"
#include <memory>
#include <fstream>

// I-O part

void usage(char *s)
{

  cerr << s
       << " <nufft_data fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double c> <X outfile>"
       << " {X initfile} {-nonneg} {-cl_box box_fname} {-maxiter N} {-eps epsilon} {-log log_fname}"
       << "\n\n";
  
  cerr << "  <nufft_data fname>:  file name of nufft_file." << endl;
  cerr << "  <double lambda_l1>:  lambda_l1.  Positive."    << endl;
  cerr << "  <double lambda_tv>:  lambda_tv.  Positive."    << endl;
  cerr << "  <double lambda_tsv>: lambda_tsv. Positive."    << endl;
  cerr << "  <double c>:          c.          Positive."    << endl;
  cerr << "  <X outfile>:         file name to write X."    << "\n\n";

  cerr << " Options." << "\n\n";

  cerr << "  {X initfile}:        file name of initial X."       << endl;
  cerr << "  {-nonneg}:           Use this if x is nonnegative." << endl;
  cerr << "  {-maxiter N}:        maximum number of iterations." << endl;
  cerr << "  {-eps epsilon}:      epsilon to check convergence." << endl;
  cerr << "  {-cl_box box_fname}: file name of CLEAN box (float)." << endl;
  cerr << "  {-log log_fname}:    log file name."                << "\n\n";

  cerr << " This solves one of the following problems with nonuniform FFT."
       << "\n\n";

  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1"  << "\n\n";

  cerr << "    or"       << "\n\n";

  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tv TV(x)"
       << "\n\n";
  
  cerr << "    or"       << "\n\n";

  cerr << " argmin |v-Ax|_2^2/2 + lambda_l1 |x|_1 + lambda_tsv TSV(x)"
       << "\n\n";

  cerr << " and write x to <X out file>" << "\n\n";

  cerr << " If {-nonneg} option is active, x is nonnegative." << "\n\n";

  cerr << " c is used for stepsize. Large c makes the algorithm" << endl;
  cerr << " stable but slow. Around 500000 is fine." << "\n\n";
  
  exit(1);
}

// commandline program

int main(int argc, char *argv[]){

  string buf_str, nufftw_fname, log_fname, init_fname, box_fname;

  int M, NN, Nx, Ny, dnum, i,
    init_flag = 0, box_flag = 0, log_flag = 0, nonneg_flag = 0,
    maxiter = MAXITER;

  float *box;

  double cinit, lambda_l1, lambda_tv, lambda_tsv, eps = EPS,
    *u_dx, *v_dy, *vis_std, *xvec, *xinit, *vis_r, *vis_i;

  struct IO_FNAMES mfista_io;
  struct RESULT    mfista_result;

  init_result(&mfista_io, &mfista_result);
  
  // check the number of variables first.

  if(argc<7) usage(argv[0]);

  // read parameters

  lambda_l1 = atof(argv[2]);
  cout << "lambda_l1  = " << lambda_l1 << endl;

  lambda_tv = atof(argv[3]);
  cout << "lambda_tv  = " << lambda_tv << endl;

  lambda_tsv = atof(argv[4]);
  cout << "lambda_tsv = " << lambda_tsv << endl;

  cinit = std::atof(argv[5]);
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
    else{
      init_flag  = 1;
      init_fname = argv[i];
    }
  }

  if(nonneg_flag == 1)
    cout << "x is nonnegative." << endl;

  if(log_flag ==1)
    cout << "Log will be saved to "
	 << "\"" << log_fname << "\"." << endl;

  cout << endl;


  // read fftw_data
  
  nufftw_fname = argv[1];

  cout << "data file name is \"" << nufftw_fname << ".\"\n";

  // open a file
  
  ifstream nufft_fs(nufftw_fname.data());

  if(nufft_fs.fail()){
    cerr << "Cannot open \"" << nufftw_fname << ".\"\n";
    exit(0);
  }

  // read data

  getline(nufft_fs, buf_str);
  if(sscanf(buf_str.data(), "M  = %d\n", &M) != 1)  exit(0);

  getline(nufft_fs, buf_str);  
  if(sscanf(buf_str.data(), "NX = %d\n", &Nx) != 1) exit(0);

  getline(nufft_fs, buf_str);  
  if(sscanf(buf_str.data(), "NY = %d\n", &Ny) != 1) exit(0);

  getline(nufft_fs, buf_str);  
  if(sscanf(buf_str.data(), "\n") != 0)             exit(0);

  getline(nufft_fs, buf_str);  
  if(sscanf(buf_str.data(),
	     "u_rad, v_rad, vis_r, vis_i, noise_std_dev\n") != 0) exit(0);

  getline(nufft_fs, buf_str);  
  if(sscanf(buf_str.data(), "\n") != 0)             exit(0);

  // print data size

  NN = Nx*Ny;
  
  cout << "number of u-v points:  " << M  << endl;
  cout << "X-dim of image:        " << Nx << endl;
  cout << "Y-dim of image:        " << Ny << endl;
  cout << "Data size Nx x Ny:     " << NN << endl;

  // allocate vectors (RAII)
  
  std::unique_ptr<double []> u_p(new double[M]);
  std::unique_ptr<double []> v_p(new double[M]);
  std::unique_ptr<double []> r_p(new double[M]);
  std::unique_ptr<double []> i_p(new double[M]);
  std::unique_ptr<double []> s_p(new double[M]);
  std::unique_ptr<double []> xi_p(new double[NN]);
  std::unique_ptr<double []> xv_p(new double[NN]);
  std::unique_ptr<float []> b_p(new float[NN]);
  u_dx = u_p.get();
  v_dy = v_p.get();
  vis_r  = r_p.get();
  vis_i  = i_p.get();
  vis_std  = s_p.get();
  xinit = xi_p.get();
  xvec  = xv_p.get();
  box = b_p.get();

  // read visibilities

  for(i = 0; i < M; i++){
    getline(nufft_fs, buf_str);  
    if(sscanf(buf_str.data(), "%le, %le, %le, %le, %le\n",
	      u_dx+i, v_dy+i, vis_r+i, vis_i+i, vis_std+i)!=5){

      cerr << "cannot read data." << endl;
      exit(0);
    }
  }

  nufft_fs.close();

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

  mfista_imaging_core_nufft(u_dx, v_dy, vis_r, vis_i, vis_std,
			    M, Nx, Ny, maxiter, eps, lambda_l1, lambda_tv, lambda_tsv, cinit,
			    xinit, xvec, nonneg_flag, box_flag, box, &mfista_result);

  // write resulting image to a file

  ofstream out_fs(argv[6], ios::out | ios::binary);
  out_fs.write((char*)xvec, sizeof(double)*NN);
  out_fs.close();

  // output results 
  
  mfista_io.fft_fname = argv[1];
  mfista_io.out_fname = argv[6];

  if(init_flag == 1) mfista_io.in_fname = (char*)init_fname.data();

  cout_result(argv[0], &mfista_io, &mfista_result);

  if(log_flag == 1){
    ofstream log_fs(log_fname.data(), ios::out);
    write_result(&log_fs, argv[0], &mfista_io, &mfista_result);
    log_fs.close();
  }
}

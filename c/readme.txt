Usage:

when A is a real matrix (M x N), and Y is a vector (M), this
sources provide the following programs.

./mfista_L1 <int M> <int N> <Y_fname> <A_fname> <double lambda_l1> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-looe} {-log logfilename}

./mfista_L1_TV <int M> <int N> <Y_fname> <A_fname> <double lambda_l1> <double lambda_tv> <double cinit> <x.out_fname> {x_init_filename} {-rec Nx} {-nonneg} {-log logfilename}

./mfista_L1_TSV <int M> <int N> <Y_fname> <A_fname> <double lambda_l1> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-rec Nx} {-nonneg} {-looe} {-log logfilename}

./mfista_imaging_dft <int M> <int N> <Y_fname> <A_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-rec Nx} {-nonneg} {-looe} {-log logfilename}
 
if the data points are on grids, you can use fft. fft implementations
are much fater and require small size of memory. the following programs
are implemented with fft.

./mfista_L1_fft <fft_fname> <double lambda_l1> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-looe} {-log logfilename}

./mfista_L1_TV_fft <fft_fname> <double lambda_l1> <double lambda_tv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-log logfilename}

./mfista_L1_TSV_fft <fft_fname> <double lambda_l1> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-looe} {-log logfilename}

./mfista_imaging_fft <fft_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-looe} {-log logfilename}

for <fft_fname> find fft_data.txt and see the data.

Options:

1. {x_init_filename}: This option specifies a initial values for
   x. If not specified, x is set to 0.

2. {-nonneg}: If x is nonnegative, use this option.

3. {-rec Nx}: For mfista_TV, mfista_L1_TV*, and mfista_L1_sqTV*. This
   option should be used when image is not square. In the above case,
   the image size is

   Nx * Ny = 10000

   where Ny is automatically computed from 10000/Nx.

   If you do not set Nx, then Nx = sqrt(N).

4. {-looe}: For mfista_L1, and mfista_L1_sqTV. If this is set,
   Obuchi & Kabashima's approximation of Leave One Out Error will be
   computed.

5. {-log logfilename}: Summary of the run is shown on the screen by
   default. If this option is specified, it will be also saved in the
   file.

Note: "mfista_imaging_dft" includes all of the other programs, but
you cannot make both of lambda_tv and lambda_tsv positive. And when
you make lambda_tv positive, you cannot use {-looe}. For further
instruction type "mfista_* --help".


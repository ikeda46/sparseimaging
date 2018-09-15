Usage:

If the data points are on grids, fft is fater and require less
memory. The following program is implemented with fft.

./mfista_imaging_fft <fft_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-maxiter ITER} {-eps EPS} {-cl_box box_filename} {-log logfilename}

for <fft_fname>   find fft_data.txt and see the data.

For non uniform grids, Nufft can be applied. The following program is implemented with nufft.

./mfista_imaging_nufft <nufft_fname> <double lambda_l1> <double lambda_tv> <double lambda_tsv> <double cinit> <x.out_fname> {x_init_filename} {-nonneg} {-maxiter ITER} {-eps EPS} {-cl_box box_filename} {-log logfilename}

for <nufft_fname> find nufft_data.txt and see the data.

Options:

1. {x_init_filename}: This option specifies a initial values for x. If
   not specified, x is set to 0.

2. {-maxiter ITER}: Maximum number of iteration. Default is 50000.

3. {-eps EPS}: EPS is used to determine the convergence. 

4. {-nonneg}: If x is nonnegative, use this option.

5. {-cl_box box_fname}: Use "box_fname file" (float) for clean
   box. This indicates the active pixels.

6. {-log logfilename}: Summary of the run is shown on the screen by
   default. If this option is specified, it will be also saved in the
   file.

Note: You cannot make both of lambda_tv and lambda_tsv positive. For
      further instruction type "mfista_* --help".


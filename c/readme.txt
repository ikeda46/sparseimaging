Usage:

when A is a real matrix (2892 x 10000), and Y is a vector (2892), this
sources provide the following programs.

./mfista_L1 2892 10000 Y.obs A.obs 1 50000 x_l1.out {x_init_filename} {-nonneg} {-looe} {-log logfilename}

./mfista_L1_TV 2892 10000 Y.obs A.obs 1 1 50000 x_l1_tv.out {x_init_filename} {-rec Nx} {-nonneg}

./mfista_L1_sqTV 2892 10000 Y.obs A.obs 1 1 50000 x_l1_sqtv.out {x_init_filename} {-rec Nx} {-nonneg} {-looe} {-log logfilename} 


Options:

1. {x_init_filename}: This option specifies a initial values for
   x. If not specified, x is set to 0.

2. {-nonneg}: If x is nonnegative, use this option.

2. {-rec Nx}: For mfista_TV, mfista_L1_TV*, and mfista_L1_sqTV*. This
   option should be used when image is not square. In the above case,
   the image size is

   Nx * Ny = 10000

   where Ny is automatically computed from 10000/Nx.

   If you do not spesify Nx, then Nx = sqrt(10000) in the above case.

3. {-looe}: For mfista_L1, and mfista_L1_sqTV. If this is specified,
   Obuchi & Kabashima's approximation of Leave One Out Error will be
   computed.

4. {-log logfilename}: Summary of the run is shown on the screen by
   default. If this option is specified, it will be also saved in the
   file.


for further instruction type "mfista_* --help".


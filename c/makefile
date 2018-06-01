# suffixes
.SUFFIXES: .o2

# INSTALLDIR
PYTHONLIBDIR=../../pythonlib
LIBDIR=../../lib
INCLUDEDIR=../../include
BINDIR=../../bin

# compile options

CC=gcc
CXX=g++

# use the following two lines to comple without PTHREAD
# CFLAGS=-O2
# CLIBS_FFTW = -lfftw3 

# use the following two lines to enable PTHREAD choose the thread_num properly.
CFLAGS=-O2 -DPTHREAD -DTHREAD_NUM=4
CXXFLAGS = -fPIC -Ofast -funroll-loops -march=native -DNEED_EXTERN_C

CLIBS_FFTW = -lfftw3_threads -lfftw3
CLIBS_NUFFTW = -fPIC -Ofast -funroll-loops -march=native -fopenmp -lfftw3 -lfftw3_threads -lstdc++ -lm


CLIBS= -lopenblas -lm -lrt -lpthread
CLIBS_LAPACK = -llapack
# set the path to the static library of finufft. they wrote
# static library works for Mac OS.
NUFFTLIB_STATIC = lib/libfinufft.a

targets = mfista_imaging_dft mfista_imaging_nufft mfista_imaging_fft
object_io = mfista_io.o
object_tools = mfista_tools.o 
object_dft = looe_lib.o mfista_TV_lib.o mfista_dft_lib.o 
object_nufft = mfista_TV_lib.o mfista_nufft_lib.o finufft2d4smili.o
object_fft = mfista_TV_lib.o mfista_fft_lib.o

object_tools2 = mfista_tools.o2 
object_dft2 = looe_lib.o2 mfista_TV_lib.o2 mfista_dft_lib.o2
object_nufft2 = mfista_TV_lib.o2 mfista_nufft_lib.o2
object_fft2 = mfista_TV_lib.o2 mfista_fft_lib.o2

all: $(targets)

mfista_imaging_dft: mfista_imaging_dft.o $(object_io) $(object_dft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_dft) $(object_tools) $@.o $(CLIBS) $(CLIBS_LAPACK) -o $@

mfista_imaging_nufft: mfista_imaging_nufft.o $(object_io) $(object_nufft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_nufft) $(object_tools) $@.o $(CLIBS) $(NUFFTLIB_STATIC) $(CLIBS_NUFFTW)  -o $@

mfista_imaging_fft: mfista_imaging_fft.o $(object_io) $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

libraries: libmfista_dft libmfista_fft libmfista_nufft

libmfista_dft: $(object_tools2) $(object_dft2)
	$(CC) $(CFLAGS) -shared $(object_dft2) $(object_tools2) -o $@.so $(CLIBS) $(CLIBS_LAPACK)

libmfista_nufft: $(object_tools2) $(object_nufft2)
	$(CC) $(CFLAGS) -shared $(object_nufft2) $(object_tools2) -o $@.so $(CLIBS) $(NUFFTLIB_STATIC) $(CLIBS_NUFFTW) -Xlinker -rpath -Xlinker /usr/lib

libmfista_fft: $(object_tools2) $(object_fft2)
	$(CC) $(CFLAGS) -shared $(object_fft2) $(object_tools2) -o $@.so $(CLIBS) $(CLIBS_FFTW) -Xlinker -rpath -Xlinker /usr/lib

.c.o:
	$(CC) -c -O2 -Wall $(CFLAGS) -o $@ $<

.c.o2:
	$(CC) -c -O2 -Wall -fPIC $(CFLAGS) -o $@ $<

.ccp.o:
	$(CXX) -c -O2 -Wall $(CXXFLAGS) -o $@ $<

.ccp.o2:
	$(CXX) -c -O2 -Wall -fPIC $(CXXFLAGS) -o $@ $<


clean:
	rm -f $(targets) *.o *.o2 *.so

install: all
	mkdir -p $(BINDIR)
	cp mfista_imaging_dft $(BINDIR)
	cp mfista_imaging_fft $(BINDIR)

uninstall: clean
	rm -f makefile
	rm -f $(BINDIR)/mfista_imaging_dft
	rm -f $(BINDIR)/mfista_imaging_fft

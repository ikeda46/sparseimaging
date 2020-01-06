# suffixes
.SUFFIXES: .o2

# INSTALLDIR
PYTHONLIBDIR=../../pythonlib
LIBDIR=../../lib
INCLUDEDIR=../../include
BINDIR=../../bin

# compile options

CC=gcc

# use the following two lines to comple without PTHREAD
# CFLAGS=-O2
# CLIBS_FFTW = -lfftw3 

# use the following two lines to enable PTHREAD choose the thread_num properly.
CFLAGS=-O2 -DPTHREAD -DTHREAD_NUM=4 -ftree-vectorize -fopt-info-all
CLIBS_FFTW = -lfftw3_threads -lfftw3

CLIBS= -lopenblas -lm -lrt -lpthread

targets = mfista_imaging_nufft mfista_imaging_fft
object_io = mfista_io.o
object_tools = mfista_tools.o 
object_nufft = mfista_TV_lib.o mfista_nufft_lib.o
object_fft = mfista_TV_lib.o mfista_fft_lib.o

object_tools2 = mfista_tools.o2 
object_nufft2 = mfista_TV_lib.o2 mfista_nufft_lib.o2
object_fft2 = mfista_TV_lib.o2 mfista_fft_lib.o2

object_dft = mfista_TV_lib.o mfista_dft_lib.o

all: $(targets)

mfista_imaging_nufft: mfista_imaging_nufft.o $(object_io) $(object_nufft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_nufft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW)  -o $@

mfista_imaging_fft: mfista_imaging_fft.o $(object_io) $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

mfista_imaging_dft: mfista_imaging_dft.o $(object_io) $(object_dft) $(object_tools)
	$(CC) $(CFLAGS) $(object_io) $(object_dft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

libraries: libmfista_fft libmfista_nufft

libmfista_nufft: $(object_tools2) $(object_nufft2)
	$(CC) $(CFLAGS) -shared $(object_nufft2) $(object_tools2) -o $@.so $(CLIBS) $(CLIBS_FFTW) -Xlinker -rpath -Xlinker /usr/lib

libmfista_fft: $(object_tools2) $(object_fft2)
	$(CC) $(CFLAGS) -shared $(object_fft2) $(object_tools2) -o $@.so $(CLIBS) $(CLIBS_FFTW) -Xlinker -rpath -Xlinker /usr/lib

.c.o:
	$(CC) -c -O2 -Wall $(CFLAGS) -o $@ $<

.c.o2:
	$(CC) -c -O2 -Wall -fPIC $(CFLAGS) -o $@ $<

clean:
	rm -f $(targets) *.o *.o2 *.so

install: all
	mkdir -p $(BINDIR)
	cp mfista_imaging_fft $(BINDIR)

uninstall: clean
	rm -f makefile
	rm -f $(BINDIR)/mfista_imaging_fft

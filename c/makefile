# suffixes
.SUFFIXES: .o2

# INSTALLDIR
PYTHONLIBDIR=../../pythonlib
LIBDIR=../../lib
INCLUDEDIR=../../include
BINDIR=../../bin

CC=gcc
CFLAGS=-O2
CLIBS= -llapack -lopenblas -lm -lrt -lpthread
CLIBS_FFTW = -lfftw3

targets = mfista_L1 mfista_L1_TV mfista_L1_TSV mfista_imaging_dft mfista_L1_fft mfista_L1_TV_fft mfista_L1_TSV_fft mfista_imaging_fft 
object_tools = mfista_tools.o looe_lib.o 
object_tv = mfista_TV_lib.o
object_tsv = mfista_TSV_lib.o
object_dft = mfista_TV_lib.o mfista_TSV_lib.o mfista_dft_lib.o
object_fft = mfista_TV_lib.o mfista_fft_lib.o

object_tools2 = mfista_tools.o2 looe_lib.o2
object_dft2 = mfista_TV_lib.o2 mfista_TSV_lib.o2 mfista_dft_lib.o2
object_fft2 = mfista_TV_lib.o2 mfista_fft_lib.o2

all: $(targets)

mfista_L1: mfista_L1.o $(object_tsv) $(object_tools)
	$(CC) $(CFLAGS) $(object_tsv) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_TV: mfista_L1_TV.o $(object_tv) $(object_tools)
	$(CC) $(CFLAGS) $(object_tv) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_TSV: mfista_L1_TSV.o $(object_tsv) $(object_tools)
	$(CC) $(CFLAGS) $(object_tsv) $(object_tools) $@.o $(CLIBS) -o $@

mfista_imaging_dft: mfista_imaging_dft.o $(object_dft) $(object_tools)
	$(CC) $(CFLAGS) $(object_dft) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_fft: mfista_L1_fft.o $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

mfista_L1_TV_fft: mfista_L1_TV_fft.o $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

mfista_L1_TSV_fft: mfista_L1_TSV_fft.o $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

mfista_imaging_fft: mfista_imaging_fft.o $(object_fft) $(object_tools)
	$(CC) $(CFLAGS) $(object_fft) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

libraries: libmfista_dft libmfista_fft

libmfista_dft: $(object_tools2) $(object_dft2)
	$(CC) $(CFLAGS) -shared $(object_dft) $(object_tools) -o $@.so $(CLIBS) $(CLIBS_FFTW)

libmfista_fft: $(object_tools2) $(object_fft2)
	$(CC) $(CFLAGS) -shared $(object_fft) $(object_tools) -o $@.so $(CLIBS) $(CLIBS_FFTW)

.c.o:
	$(CC) -c -O2 -Wall $(CFLAGS) -o $@ $<

.c.o2:
	$(CC) -c -O2 -Wall -fPIC $(CFLAGS) -o $@ $<

clean:
	rm -f $(targets) *.o *.o2 *.so

install: all
	mkdir -p $(BINDIR)
	cp mfista_L1 $(BINDIR)
	cp mfista_L1_TV $(BINDIR)
	cp mfista_L1_TSV $(BINDIR)

uninstall: clean
	rm -f makefile
	rm -f $(BINDIR)/mfista_L1
	rm -f $(BINDIR)/mfista_L1_TV
	rm -f $(BINDIR)/mfista_L1_TSV

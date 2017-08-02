# INSTALLDIR
PYTHONLIBDIR=../../pythonlib
LIBDIR=../../lib
INCLUDEDIR=../../include
BINDIR=../../bin

CC=gcc
CFLAGS=-O2
CLIBS= -llapack -lopenblas -lm -lpthread
CLIBS_FFTW = -lfftw3

targets = mfista_L1 mfista_L1_TV mfista_L1_TSV mfista_L1_TSV_fftw
object_tools = mfista_tools.o
object_l1 = mfista_lib.o
object_tv = mfista_TV_lib.o
object_tsv = mfista_TSV_lib.o
object_fft = mfista_fftw_lib.o

all: $(targets)

mfista_L1: mfista_L1.o $(object_l1) $(object_tools)
	$(CC) $(CFLAGS) $(object_l1) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_TV: mfista_L1_TV.o $(object_tv) $(object_tools)
	$(CC) $(CFLAGS) $(object_tv) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_TSV: mfista_L1_TSV.o $(object_tsv) $(object_tools)
	$(CC) $(CFLAGS) $(object_tsv) $(object_tools) $@.o $(CLIBS) -o $@

mfista_L1_TSV_fftw: mfista_L1_TSV_fftw.o $(object_fft) $(object_tsv) $(object_tools)
	$(CC) $(CFLAGS) $(object_fft) $(object_tsv) $(object_tools) $@.o $(CLIBS) $(CLIBS_FFTW) -o $@

.c.o:
	$(CC) -c -O2 -Wall $(CFLAGS) -o $@ $<

clean:
	rm -f $(targets) *.o

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

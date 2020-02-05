# suffixes
.SUFFIXES: .o2

# INSTALLDIR
PYTHONLIBDIR=../../pythonlib
LIBDIR=../../lib
EIGENLIBRARY= /usr/include/eigen3
INCLUDEDIR=../../include
BINDIR=../../bin

# compile options

CXX=g++
CXX_VERSION= -std=c++17

# use the following two lines to comple without PTHREAD
# CFLAGS=-O2
# CLIBS_FFTW = -lfftw3 

# use the following two lines to enable PTHREAD choose the thread_num properly.
CFLAGS=-O2 -DPTHREAD -DTHREAD_NUM=4 -I/usr/include/eigen3/
CLIBS_FFTW = -lfftw3_threads -lfftw3

CLIBS= -lm -lrt -lpthread

targets = mfista_imaging_nufft mfista_imaging_fft
object_io = mfista_io.o
object_tools = mfista_tools.o
object_memory = mfista_memory.o
object_fft = mfista_fft_lib.o
object_nufft = mfista_nufft_lib.o

object_tools2 = mfista_tools.o2
object_memory = mfista_memory.o2
object_fft2 = mfista_fft_lib.o2
object_nufft2 = mfista_nufft_lib.o2

all: $(targets)

mfista_imaging_nufft: mfista_imaging_nufft.o $(object_io) $(object_nufft) $(object_tools) $(object_memory)
	$(CXX) ${CXX_VERSION} $(CFLAGS) $(object_io) $(object_nufft) $(object_tools) $(object_memory) $@.o $(CLIBS) $(CLIBS_FFTW)  -o $@

mfista_imaging_fft: mfista_imaging_fft.o $(object_io) $(object_fft) $(object_tools) $(object_memory)
	$(CXX) ${CXX_VERSION} $(CFLAGS) $(object_io) $(object_fft) $(object_tools) $(object_memory) $@.o $(CLIBS) $(CLIBS_FFTW)  -o $@

libraries: libmfista_nufft libmfista_fft

libmfista_nufft: $(object_tools2) $(object_nufft2) $(object_memory2)
	$(CXX) ${CXX_VERSION} $(CFLAGS) -shared $(object_nufft2) $(object_tools2) $(object_memory2) -o $@.so $(CLIBS) $(CLIBS_FFTW) -I${EIGENLIBRARY} -Xlinker -rpath -Xlinker /usr/lib

libmfista_fft: $(object_tools2) $(object_fft2) $(object_memory2)
	$(CXX) ${CXX_VERSION} $(CFLAGS) -shared $(object_fft2) $(object_tools2) $(object_memory2) -o $@.so $(CLIBS) $(CLIBS_FFTW) -I${EIGENLIBRARY} -Xlinker -rpath -Xlinker /usr/lib

.cpp.o:
	$(CXX) ${CXX_VERSION} -c -O2 -Wall $(CFLAGS) -I${EIGENLIBRARY} -fPIC -o $@ $<

.cpp.o2:
	$(CXX) ${CXX_VERSION} -c -O2 -Wall $(CFLAGS) -I${EIGENLIBRARY} -fPIC -o $@ $<

clean:
	rm -f $(targets) *.o *.o2 *.so

install: all
	mkdir -p $(BINDIR)
	cp mfista_imaging_fft $(BINDIR)

uninstall: clean
	rm -f makefile
	rm -f $(BINDIR)/mfista_imaging_fft

#include "mfista_memory.hpp"

#include <functional>
#include <new>

// utility for memory management
void *allocate_fftw_complex(size_t n) {
  void *p = fftw_malloc(n);
  if (p == NULL) {
    throw std::bad_alloc();
  }
  return p;
}

void deallocate_fftw_complex(void *ptr) {
  if (ptr != nullptr) {
    fftw_free(ptr);
  }
}

void deallocate_fftw_plan(fftw_plan ptr) {
  if (ptr != nullptr) {
    fftw_destroy_plan(ptr);
  }
}



#ifndef _MFISTA_MEMORY_HPP_
#define _MFISTA_MEMORY_HPP_

#include <fftw3.h>
#include <functional>

void *allocate_fftw_complex(size_t n);

void deallocate_fftw_complex(void *ptr);

void deallocate_fftw_plan(fftw_plan ptr);

class ScopeGuard {
  typedef std::function<void(void) noexcept> Func;
public:
  ScopeGuard() = delete;
  explicit ScopeGuard(Func clean_up, bool enabled = true) :
  clean_up_(clean_up), engaged_(enabled) {
  }
  ScopeGuard(ScopeGuard const &other) = delete;
  ScopeGuard &operator =(ScopeGuard const &other) = delete;
  ScopeGuard const &operator =(ScopeGuard const &other) const = delete;
  void *operator new(std::size_t) = delete;
  ~ScopeGuard() {
    if (engaged_) {
      clean_up_();
    }
  }
  void Disable() {
    engaged_ = false;
  }
  void CleanUpNow() {
    if (engaged_) {
      clean_up_();
      engaged_ = false;
    }
  }
private:
        Func clean_up_;
        bool engaged_;
};

#endif

#ifndef CROSSPY_TIMER_H
#define CROSSPY_TIMER_H

#include <chrono>
#include <string>

// using Str = std::basic_string<char, std::char_traits<char>, Pow2Alloc<char>>;
using Str = std::string;


// #include "galois/config.h"
// #include "galois/gstl.h"

//! A simple timer
class Timer {
  typedef std::chrono::steady_clock clockTy;
  // typedef std::chrono::high_resolution_clock clockTy;
  std::chrono::time_point<clockTy> startT, stopT;

public:
  void start();
  void stop();
  uint64_t get() const;
  uint64_t get_usec() const;
};

//! A multi-start time accumulator.
//! Gives the final runtime for a series of intervals
class TimeAccumulator {
  Timer ltimer;
  uint64_t acc;

public:
  TimeAccumulator();

  void start();
  //! adds the current timed interval to the total
  void stop();
  uint64_t get() const;
  uint64_t get_usec() const;
  TimeAccumulator& operator+=(const TimeAccumulator& rhs);
  TimeAccumulator& operator+=(const Timer& rhs);
};

//! Galois Timer that automatically reports stats upon destruction
//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  Str name_;
  Str region_;
  bool valid_;

public:
  StatTimer(const char* name, const char* region);

  StatTimer(const char* const n) : StatTimer(n, nullptr) {}

  StatTimer() : StatTimer(nullptr, nullptr) {}

  StatTimer(const StatTimer&) = delete;
  StatTimer(StatTimer&&)      = delete;
  StatTimer& operator=(const StatTimer&) = delete;
  StatTimer& operator=(StatTimer&&) = delete;

  ~StatTimer();

  void start();
  void stop();
  uint64_t get_usec() const;
};

template <bool Enable>
class CondStatTimer : public StatTimer {
public:
  CondStatTimer(const char* const n, const char* region)
      : StatTimer(n, region) {}

  CondStatTimer(const char* region) : CondStatTimer("Time", region) {}
};

template <>
class CondStatTimer<false> {
public:
  CondStatTimer(const char*) {}
  CondStatTimer(const char* const, const char*) {}

  void start() const {}
  void stop() const {}
  uint64_t get_usec() const { return 0; }
};

template <typename F>
void timeThis(const F& f, const char* const name) {
  StatTimer t("Time", name);

  t.start();

  f();

  t.stop();
}

#endif

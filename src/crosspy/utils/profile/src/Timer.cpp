#include "Timer.h"
// #include "galois/runtime/Statistics.h"

void Timer::start() { startT = clockTy::now(); }

void Timer::stop() { stopT = clockTy::now(); }

uint64_t Timer::get() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(stopT - startT)
      .count();
}

uint64_t Timer::get_usec() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(stopT - startT)
      .count();
}

TimeAccumulator::TimeAccumulator() : ltimer(), acc(0) {}

void TimeAccumulator::start() { ltimer.start(); }

void TimeAccumulator::stop() {
  ltimer.stop();
  acc += ltimer.get_usec();
}

uint64_t TimeAccumulator::get() const { return acc / 1000; }
uint64_t TimeAccumulator::get_usec() const { return acc; }

TimeAccumulator& TimeAccumulator::operator+=(const TimeAccumulator& rhs) {
  acc += rhs.acc;
  return *this;
}

TimeAccumulator& TimeAccumulator::operator+=(const Timer& rhs) {
  acc += rhs.get_usec();
  return *this;
}

StatTimer::StatTimer(const char* const name, const char* const region) {
  const char* n = name ? name : "Time";
  const char* r = region ? region : "(NULL)";

  name_   = Str(n);
  region_ = Str(r);

  valid_ = false;
}

StatTimer::~StatTimer() {
  if (valid_) {
    stop();
  }

  // only report non-zero stat
  // if (TimeAccumulator::get()) {
  //   galois::runtime::reportStat_Tmax(region_, name_, TimeAccumulator::get());
  // }
}

void StatTimer::start() {
  TimeAccumulator::start();
  valid_ = true;
}

void StatTimer::stop() {
  valid_ = false;
  TimeAccumulator::stop();
}

uint64_t StatTimer::get_usec() const { return TimeAccumulator::get_usec(); }

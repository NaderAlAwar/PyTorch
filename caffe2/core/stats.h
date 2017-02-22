#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "caffe2/core/logging.h"
#include "caffe2/core/static_tracepoint.h"

namespace caffe2 {

class StatValue {
  std::atomic<int64_t> v_{0};

 public:
  int64_t update(int64_t inc) {
    return v_ += inc;
  }

  int64_t reset(int64_t value = 0) {
    return v_.exchange(value);
  }

  int64_t get() const {
    return v_.load();
  }
};

struct ExportedStatValue {
  std::string key;
  int64_t value;
  std::chrono::time_point<std::chrono::high_resolution_clock> ts;
};

/**
 * @brief Holds names and values of counters exported from a StatRegistry.
 */
using ExportedStatList = std::vector<ExportedStatValue>;
using ExportedStatMap = std::unordered_map<std::string, int64_t>;

ExportedStatMap toMap(const ExportedStatList& stats);

/**
 * @brief Holds a map of atomic counters keyed by name.
 *
 * The StatRegistry singleton, accessed through StatRegistry::get(), holds
 * counters registered through the macro CAFFE_EXPORTED_STAT. Example of usage:
 *
 * struct MyCaffeClass {
 *   MyCaffeClass(const std::string& instanceName): stats_(instanceName) {}
 *   void run(int numRuns) {
 *     try {
 *       CAFFE_EVENT(stats_, num_runs, numRuns);
 *       tryRun(numRuns);
 *       CAFFE_EVENT(stats_, num_successes);
 *     } catch (std::exception& e) {
 *       CAFFE_EVENT(stats_, num_failures, 1, "arg_to_usdt", e.what());
 *     }
 *     CAFFE_EVENT(stats_, usdt_only, 1, "arg_to_usdt");
 *   }
 *  private:
 *   struct MyStats {
 *     CAFFE_STAT_CTOR(MyStats);
 *     CAFFE_EXPORTED_STAT(num_runs);
 *     CAFFE_EXPORTED_STAT(num_successes);
 *     CAFFE_EXPORTED_STAT(num_failures);
 *     CAFFE_STAT(usdt_only);
 *   } stats_;
 * };
 *
 * int main() {
 *   MyCaffeClass a("first");
 *   MyCaffeClass b("second");
 *   for (int i = 0; i < 10; ++i) {
 *     a.run(10);
 *     b.run(5);
 *   }
 *   ExportedStatList finalStats;
 *   StatRegistry::get().publish(finalStats);
 * }
 *
 * For every new instance of MyCaffeClass, a new counter is created with
 * the instance name as prefix. Everytime run() is called, the corresponding
 * counter will be incremented by the given value, or 1 if value not provided.
 *
 * Counter values can then be exported into an ExportedStatList. In the
 * example above, considering "tryRun" never throws, `finalStats` will be
 * populated as follows:
 *
 *   first/num_runs       100
 *   first/num_successes   10
 *   first/num_failures     0
 *   second/num_runs       50
 *   second/num_successes  10
 *   second/num_failures    0
 *
 * The event usdt_only is not present in ExportedStatList because it is declared
 * as CAFFE_STAT, which does not create a counter.
 *
 * Additionally, for each call to CAFFE_EVENT, a USDT probe is generated.
 * The probe will be set up with the following arguments:
 *   - Probe name: field name (e.g. "num_runs")
 *   - Arg #0: instance name (e.g. "first", "second")
 *   - Arg #1: For CAFFE_EXPORTED_STAT, value of the updated counter
 *             For CAFFE_STAT, -1 since no counter is available
 *   - Args ...: Arguments passed to CAFFE_EVENT, including update value
 *             when provided.
 *
 * It is also possible to create additional StatRegistry instances beyond
 * the singleton. These instances are not automatically populated with
 * CAFFE_EVENT. Instead, they can be populated from an ExportedStatList
 * structure by calling StatRegistry::update().
 *
 */
class StatRegistry {
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<StatValue>> stats_;

 public:
  /**
   * Retrieve the singleton StatRegistry, which gets populated
   * through the CAFFE_EVENT macro.
   */
  static StatRegistry& get();

  /**
   * Add a new counter with given name. If a counter for this name already
   * exists, returns a pointer to it.
   */
  StatValue* add(const std::string& name);

  /**
   * Populate an ExportedStatList with current counter values.
   * If `reset` is true, resets all counters to zero. It is guaranteed that no
   * count is lost.
   */
  void publish(ExportedStatList& exported, bool reset = false);

  ExportedStatList publish(bool reset = false) {
    ExportedStatList stats;
    publish(stats, reset);
    return stats;
  }

  /**
   * Update values of counters contained in the given ExportedStatList to
   * the values provided, creating counters that don't exist.
   */
  void update(const ExportedStatList& data);

  ~StatRegistry();
};

struct Stat {
  std::string groupName;
  std::string name;
  Stat(const std::string& gn, const std::string& n) : groupName(gn), name(n) {}

  template <typename... Unused>
  int64_t operator()(Unused...) {
    return -1;
  }
};

class ExportedStat : public Stat {
  StatValue* value_;

 public:
  ExportedStat(const std::string& gn, const std::string& n)
      : Stat(gn, n), value_(StatRegistry::get().add(gn + "/" + n)) {}

  template <typename... Unused>
  int64_t operator()(int64_t increment, Unused...) {
    return value_->update(increment);
  }

  int64_t operator()() {
    return operator()(1);
  }
};

#define CAFFE_STAT_CTOR(ClassName)                 \
  ClassName(std::string name) : groupName(name) {} \
  std::string groupName

#define CAFFE_EXPORTED_STAT(name) \
  ExportedStat name {             \
    groupName, #name              \
  }

#define CAFFE_STAT(name) \
  Stat name {            \
    groupName, #name     \
  }

#define CAFFE_EVENT(stats, field, ...)                    \
  {                                                       \
    auto __caffe_event_value_ = stats.field(__VA_ARGS__); \
    CAFFE_SDT(                                            \
        field,                                            \
        stats.field.groupName.c_str(),                    \
        __caffe_event_value_,                             \
        ##__VA_ARGS__);                                   \
  }
}

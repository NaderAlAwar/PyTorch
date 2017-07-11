#pragma once

#include "torch/csrc/autograd/ir.h"

#include <memory>
#include <vector>

namespace torch { namespace autograd {

class TracingState {
private:
  int next_unique;
  std::unique_ptr<LetBuilder> builder;
  local_list params;
public:
  TracingState()
    : next_unique(0)
    , builder(std::unique_ptr<LetBuilder>(new LetBuilder()))
    {}
  std::shared_ptr<Local> makeLocal();
  void addParam(std::shared_ptr<Local> param) { params.emplace_back(param); }
  void addBinding(local_list lvals, std::shared_ptr<Instruction> rval);
  std::shared_ptr<Graph> graph(local_list locals);
};

// Ugh, global state
extern std::unique_ptr<TracingState> GlobalTracingState;

void Tracer_enter();
std::shared_ptr<Graph> Tracer_exit(local_list locals);

}}

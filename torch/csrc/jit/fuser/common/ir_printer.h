#pragma once

#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_API IRPrinter : public IterVisitor{
public:
  IRPrinter(std::ostream& os) : 
    IterVisitor(),
    irstream_(os),
    cnt_(0)   { } 
  
  void print(const Fusion* const fusion);
  
  void handle(Statement* s) { 
    Statement::dispatch(this, s); 
  }
  void handle(Expr* e) override {
    irstream_ << "\t" << cnt_++; 
    Expr::dispatch(this, e); 
    irstream_ << "\n"; 
  }
  void handle(Val* v) override { 
    Val::dispatch(this, v); 
  }
  void handle(Float* val) override { 
    if (val->isSymbolic()) {
      irstream_ << "%f" << val->name();
    } else {
      irstream_ << *(val->value()) << "f";
    }
  }
  void handle(TensorDomain*) override {}
  void handle(TensorView*) override {}
  void handle(IterDomain*) override {}
  void handle(Tensor*) override {}

  void handle(Int*) override {}

  void handle(UnaryOp*) override {}
  void handle(BinaryOp*) override {}
  void handle(Split*) override {}
  void handle(Merge*) override {}
  void handle(Reorder*) override {}

protected:                             
  std::ostream& irstream_;
  int cnt_;
};             

} // namespace fuser
} // namespace jit
} // namespace torch

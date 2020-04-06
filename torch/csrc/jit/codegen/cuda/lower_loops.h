#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

struct UnrollPass : public OptOutMutator {
 private:
  Fusion* fusion_;
  std::vector<Expr*> lowered_exprs;
  const std::vector<Expr*>& incoming_exprs_;
  Expr* active_scope = nullptr;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Custom dispatch for Expr, want to find out of it's a TV op
  Statement* mutate(Expr*) final;

  // Open the for loop.
  Statement* mutate(ForLoop*) final;

  // Remake operations with TensorIndex
  Statement* mutate(UnaryOp*) final;
  Statement* mutate(BinaryOp*) final;

  UnrollPass(Fusion* _fusion, const std::vector<Expr*>& _incoming_exprs)
      : fusion_(_fusion), incoming_exprs_(_incoming_exprs) {}

  void runPass();

 public:
  static std::vector<Expr*> runPass(Fusion* fusion, std::vector<Expr*> exprs) {
    FusionGuard fg(fusion);
    UnrollPass up(fusion, exprs);
    up.runPass();
    return up.lowered_exprs;
  }
};

struct TORCH_CUDA_API LoopNestGenerator : public OptOutDispatch {
 private:
  std::vector<Expr*> lowered_exprs;
  Fusion* fusion_;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Active IfThenElse or ForLoop
  Expr* active_scope = nullptr;

  // Get Register allocation statement for tensorview
  Allocate* getAlloc(TensorView*);

  // Clear out the last recorded computeAtView
  void clearActiveView();
  // Set active views from computeAtView
  void setActiveView(const TensorView* const);

  // Open a new inner most for loop
  void openFor(IterDomain*);

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Update for loop structure based on this TensorView
  void updateLoopNest(TensorView*);

  // Check if a TV op, generate for loop nest around it
  void handle(Expr*) final;

  // Generate the loop nest structure and place it in lowered_exprs
  void generate();

  LoopNestGenerator(Fusion* _fusion) : fusion_(_fusion) {}

 public:
  static std::vector<Expr*> getLoopNest(Fusion* fusion) {
    FusionGuard fg(fusion);
    LoopNestGenerator lng(fusion);
    lng.generate();
    return lng.lowered_exprs;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
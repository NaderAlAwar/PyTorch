#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/runtime/graph_executor.h"
#include "torch/jit.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace torch {
namespace jit {

TEST(GraphExecutorTest, Basic_CUDA) {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto g = build_lstm();
  GraphExecutor executor(g, "");
  auto stack = createStack({input, hx, cx, w_ih, w_hh});
  executor.run(stack);
  ASSERT_EQ(stack.size(), 2);
  auto [r0, r1] = lstm(input, hx, cx, w_ih, w_hh);
  ASSERT_TRUE(almostEqual(stack[0].toTensor(), r0));
  ASSERT_TRUE(almostEqual(stack[1].toTensor(), r1));
}

TEST(GraphExecutorTest, runAsync_executor) {
  /*
  TODO: there are some problem with C++ parsing script program involving
  fork. Use the test module below for now.
  issue about this: github.com/pytorch/pytorch/issues/46368
  The test module file is generated by following:
    class DemoModule(torch.nn.Module):
      def forward(self):
        r1 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        r2 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        return r1.wait() + r2.wait()
  demo = DemoModule()
  torch.jit.save(torch.jit.script(demo), 'test_interpreter_async.pt')
  */
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("test_interpreter_async.pt");
  auto module = load(testModelFile);
  auto graph = module.get_method("forward").graph();
  GraphExecutor graphExecutor(graph, "");
  auto asyncCounter = 0;
  std::mutex mtx;
  // a dummy executor which actually use at::launch, but add up a counter
  auto launcher = [&](std::function<void()> f) {
    mtx.lock();
    ++asyncCounter;
    mtx.unlock();
    at::launch(std::move(f));
  };
  std::vector<IValue> stack;
  // NOLINTNEXTLINE(modernize-use-emplace)
  stack.push_back(module._ivalue());
  graphExecutor.runAsync(stack, launcher)->wait();
  ASSERT_TRUE(asyncCounter > 0);
}

} // namespace jit
} // namespace torch

#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

clone_executorch() {
  EXECUTORCH_PINNED_COMMIT=$(get_pinned_commit executorch)

  # Clone the Executorch
  git clone https://github.com/pytorch/executorch.git

  # and fetch the target commit
  pushd executorch
  git checkout "${EXECUTORCH_PINNED_COMMIT}"
  git submodule update --init
  popd

  chown -R jenkins executorch
}

install_buck2() {
  pushd executorch/.ci/docker

  BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)
  source common/install_buck.sh

  popd
}

install_conda_dependencies() {
  pushd executorch/.ci/docker
  # Install conda dependencies like flatbuffer
  conda_install --file conda-env-ci.txt
  popd
}

install_pip_dependencies() {
  pushd executorch
  as_jenkins bash install_requirements.sh --pybind xnnpack
  popd
}

setup_executorch() {
  pushd executorch
  # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
  as_jenkins bash .ci/scripts/setup-vulkan-linux-deps.sh

  # Build ExecuTorch later after we install PyTorch built from source
  popd
}

clone_executorch
install_buck2
install_conda_dependencies
install_pip_dependencies
setup_executorch

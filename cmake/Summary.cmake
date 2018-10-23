# Prints accumulated Caffe2 configuration summary
function (caffe2_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  CMake version         : ${CMAKE_VERSION}")
  message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  BLAS                  : ${BLAS}")
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions   : ${tmp}")
  message(STATUS "  CMAKE_PREFIX_PATH     : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX  : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "")

  message(STATUS "  TORCH_VERSION         : ${TORCH_VERSION}")
  message(STATUS "  CAFFE2_VERSION        : ${CAFFE2_VERSION}")
  message(STATUS "  BUILD_ATEN_MOBILE     : ${BUILD_ATEN_MOBILE}")
  message(STATUS "  BUILD_ATEN_ONLY       : ${BUILD_ATEN_ONLY}")
  message(STATUS "  BUILD_BINARY          : ${BUILD_BINARY}")
  message(STATUS "  BUILD_CUSTOM_PROTOBUF : ${BUILD_CUSTOM_PROTOBUF}")
  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    message(STATUS "    Link local protobuf : ${CAFFE2_LINK_LOCAL_PROTOBUF}")
  else()
    message(STATUS "    Protobuf compiler   : ${PROTOBUF_PROTOC_EXECUTABLE}")
    message(STATUS "    Protobuf includes   : ${PROTOBUF_INCLUDE_DIRS}")
    message(STATUS "    Protobuf libraries  : ${PROTOBUF_LIBRARIES}")
  endif()
  message(STATUS "  BUILD_DOCS            : ${BUILD_DOCS}")
  message(STATUS "  BUILD_PYTHON          : ${BUILD_PYTHON}")
  if (${BUILD_PYTHON})
    message(STATUS "    Python version      : ${PYTHON_VERSION_STRING}")
    message(STATUS "    Python executable   : ${PYTHON_EXECUTABLE}")
    message(STATUS "    Pythonlibs version  : ${PYTHONLIBS_VERSION_STRING}")
    message(STATUS "    Python library      : ${PYTHON_LIBRARIES}")
    message(STATUS "    Python includes     : ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "    Python site-packages: ${PYTHON_SITE_PACKAGES}")
  endif()
  message(STATUS "  BUILD_CAFFE2_OPS      : ${BUILD_CAFFE2_OPS}")
  message(STATUS "  BUILD_SHARED_LIBS     : ${BUILD_SHARED_LIBS}")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")

  message(STATUS "  USE_ASAN              : ${USE_ASAN}")
  message(STATUS "  USE_CUDA              : ${USE_CUDA}")
  if(${USE_CUDA})
    message(STATUS "    CUDA static link    : ${CAFFE2_STATIC_LINK_CUDA}")
    message(STATUS "    USE_CUDNN           : ${USE_CUDNN}")
    message(STATUS "    CUDA version        : ${CUDA_VERSION}")
    if(${USE_CUDNN})
      message(STATUS "    cuDNN version       : ${CUDNN_VERSION}")
    endif()
    message(STATUS "    CUDA root directory : ${CUDA_TOOLKIT_ROOT_DIR}")
    get_target_property(__tmp caffe2::cuda IMPORTED_LOCATION)
    message(STATUS "    CUDA library        : ${__tmp}")
    get_target_property(__tmp caffe2::cudart INTERFACE_LINK_LIBRARIES)
    message(STATUS "    cudart library      : ${__tmp}")
    get_target_property(__tmp caffe2::cublas INTERFACE_LINK_LIBRARIES)
    message(STATUS "    cublas library      : ${__tmp}")
    get_target_property(__tmp caffe2::cufft INTERFACE_LINK_LIBRARIES)
    message(STATUS "    cufft library       : ${__tmp}")
    get_target_property(__tmp caffe2::curand IMPORTED_LOCATION)
    message(STATUS "    curand library      : ${__tmp}")
    if(${USE_CUDNN})
      get_target_property(__tmp caffe2::cudnn IMPORTED_LOCATION)
      message(STATUS "    cuDNN library       : ${__tmp}")
    endif()
    get_target_property(__tmp caffe2::nvrtc IMPORTED_LOCATION)
    message(STATUS "    nvrtc               : ${__tmp}")
    message(STATUS "    CUDA include path   : ${CUDA_INCLUDE_DIRS}")
    message(STATUS "    NVCC executable     : ${CUDA_NVCC_EXECUTABLE}")
    message(STATUS "    CUDA host compiler  : ${CUDA_HOST_COMPILER}")
    message(STATUS "    USE_TENSORRT        : ${USE_TENSORRT}")
    if(${USE_TENSORRT})
      message(STATUS "      TensorRT runtime library: ${TENSORRT_LIBRARY}")
      message(STATUS "      TensorRT include path   : ${TENSORRT_INCLUDE_DIR}")
    endif()
  endif()
  message(STATUS "  USE_ROCM              : ${USE_ROCM}")
  message(STATUS "  USE_EIGEN_FOR_BLAS    : ${CAFFE2_USE_EIGEN_FOR_BLAS}")
  message(STATUS "  USE_FFMPEG            : ${USE_FFMPEG}")
  message(STATUS "  USE_GFLAGS            : ${USE_GFLAGS}")
  message(STATUS "  USE_GLOG              : ${USE_GLOG}")
  message(STATUS "  USE_LEVELDB           : ${USE_LEVELDB}")
  if(${USE_LEVELDB})
    message(STATUS "    LevelDB version     : ${LEVELDB_VERSION}")
    message(STATUS "    Snappy version      : ${Snappy_VERSION}")
  endif()
  message(STATUS "  USE_LITE_PROTO        : ${USE_LITE_PROTO}")
  message(STATUS "  USE_LMDB              : ${USE_LMDB}")
  if(${USE_LMDB})
    message(STATUS "    LMDB version        : ${LMDB_VERSION}")
  endif()
  message(STATUS "  USE_METAL             : ${USE_METAL}")
  message(STATUS "  USE_MKL               : ${USE_MKL}")
  if(${USE_MKL})
    message(STATUS "    USE_MKLML           : ${USE_MKLML}")
    message(STATUS "    USE_IDEEP           : ${USE_IDEEP}")
  endif()
  message(STATUS "  USE_MOBILE_OPENGL     : ${USE_MOBILE_OPENGL}")
  message(STATUS "  USE_NCCL              : ${USE_NCCL}")
  if(${USE_NCCL})
    message(STATUS "    USE_SYSTEM_NCCL     : ${USE_SYSTEM_NCCL}")
  endif()
  message(STATUS "  USE_NNPACK            : ${USE_NNPACK}")
  message(STATUS "  USE_NUMPY             : ${USE_NUMPY}")
  message(STATUS "  USE_OBSERVERS         : ${USE_OBSERVERS}")
  message(STATUS "  USE_OPENCL            : ${USE_OPENCL}")
  message(STATUS "  USE_OPENCV            : ${USE_OPENCV}")
  if(${USE_OPENCV})
    message(STATUS "    OpenCV version      : ${OpenCV_VERSION}")
  endif()
  message(STATUS "  USE_OPENMP            : ${USE_OPENMP}")
  message(STATUS "  USE_PROF              : ${USE_PROF}")
  message(STATUS "  USE_REDIS             : ${USE_REDIS}")
  message(STATUS "  USE_ROCKSDB           : ${USE_ROCKSDB}")
  message(STATUS "  USE_ZMQ               : ${USE_ZMQ}")
  message(STATUS "  USE_DISTRIBUTED       : ${USE_DISTRIBUTED}")
  if(${USE_DISTRIBUTED})
    message(STATUS "    USE_MPI             : ${USE_MPI}")
    message(STATUS "    USE_GLOO            : ${USE_GLOO}")
    message(STATUS "    USE_GLOO_IBVERBS    : ${USE_GLOO_IBVERBS}")
  endif()

  message(STATUS "  Public Dependencies  : ${Caffe2_PUBLIC_DEPENDENCY_LIBS}")
  message(STATUS "  Private Dependencies : ${Caffe2_DEPENDENCY_LIBS}")
endfunction()

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.cu"
#else

THC_API int THCTensor_(getDevice)(THCState* state, const THCTensor* tensor) {
  return THCTensor_getDevice(state, tensor);
}

#endif

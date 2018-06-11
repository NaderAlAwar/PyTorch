#include "THCGeneral.h"
#include "THCTensor.hpp"
#include "THCTensorCopy.h"

#include <new>

#include "generic/THCTensor.cpp"
#include "THCGenerateAllTypes.h"

#include "THCTensorInfo.cuh"

int THCTensor_nDimension(THCState *state, const THCTensor *self) {
  return self->_dim();
}

int64_t THCTensor_size(THCState *state, const THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->_dim()), 2, "out of range");
  return self->size[dim];
}

int64_t THCTensor_stride(THCState *state, const THCTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->_dim()), 2, "out of range");
  return self->stride[dim];
}
THLongStorage *THCTensor_newSizeOf(THCState *state, THCTensor *self) {
  THLongStorage *size = THLongStorage_newWithSize(self->_dim());
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THCTensor *THCTensor_new(THCState *state, at::ScalarType scalar_type) {
  switch(scalar_type) {
    case at::ScalarType::Byte:
      return THCudaByteTensor_new(state);
    case at::ScalarType::Char:
      return THCudaCharTensor_new(state);
    case at::ScalarType::Short:
      return THCudaShortTensor_new(state);
    case at::ScalarType::Int:
      return THCudaIntTensor_new(state);
    case at::ScalarType::Long:
      return THCudaLongTensor_new(state);
#ifdef CUDA_HALF_TENSOR
    case at::ScalarType::Half:
      return THCudaHalfTensor_new(state);
#endif
    case at::ScalarType::Float:
      return THCudaTensor_new(state);
    case at::ScalarType::Double:
      return THCudaDoubleTensor_new(state);
    default:
      AT_ERROR("unexpected ScalarType: ", at::toString(scalar_type));
  }
}

void THCTensor_resizeLegacy(THCState *state, THCTensor *self, THLongStorage *size, THLongStorage *stride) {
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THCTensor_resizeNdLegacy(state, self, size->size, THLongStorage_data(size), (stride ? THLongStorage_data(stride) : NULL));
}

void THCTensor_resizeAs(THCState *state, THCTensor *self, THCTensor *src) {
  int isSame = 0;
  int d;
  if(self->_dim() == src->_dim())
  {
    isSame = 1;
    for(d = 0; d < self->_dim(); d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCTensor_resizeNdLegacy(state, self, src->_dim(), src->size, NULL);
}

void THCTensor_resizeNdLegacy(THCState *state, THCTensor *self, int nDimension, int64_t *size, int64_t *stride)
{
  int d;
  int nDimension_;
  ptrdiff_t totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->_dim() > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->_dim() > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->_dim())
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->_dim())
    {
      self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*nDimension);
      self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*nDimension);
      self->dim_ = nDimension;
    }

    totalSize = 1;
    // note: can't use _dim() here because there is junk in size
    for(d = nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        THError("Tensor: invalid null storage");
      if(totalSize+self->storageOffset > self->storage->size)
        THCStorage_resize(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else {
    self->dim_ = 1;
    self->size = (int64_t *)THRealloc(self->size, sizeof(int64_t));
    self->stride = (int64_t *)THRealloc(self->stride, sizeof(int64_t));
    self->size[0] = 0;
    self->stride[0] = 1;
  }
}

void THCTensor_set(THCState *state, THCTensor *self, THCTensor *src)
{
  if(self != src)
    THCTensor_setStorageNd(state,
                           self,
                           src->storage,
                           src->storageOffset,
                           src->_dim(),
                           src->size,
                           src->stride);
}

void THCTensor_setStorageNd(THCState *state, THCTensor *self, THCStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if (!self->storage) {
      THError("Tensor: invalid null storage");
    }
    auto scalar_type = self->storage->scalar_type;
    THCStorage_free(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THCStorage_retain(state, self->storage);
    }
    else
      self->storage = THCStorage_new(state, scalar_type);
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THCTensor_resizeNdLegacy(state, self, nDimension, size, stride);
}


void THCTensor_squeeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->dim(), 3, "dimension out of range");

  THCTensor_set(state, self, src);

#ifdef TH_SCALAR
  if(src->size[dimension] == 1)
#else
  if(src->size[dimension] == 1 && src->dim() > 1)
#endif
  {
    for(d = dimension; d < self->dim()-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->dim_--;
  }
}

void THCTensor_unsqueeze1d(THCState *state, THCTensor *self, THCTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 3, "dimension out of range");
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(!src->is_empty(), 3, "cannot unsqueeze empty tensor");
#endif

  THCTensor_set(state, self, src);

  self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*(self->dim()+1));
  self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*(self->dim()+1));
  self->dim_++;
  for (d = self->dim()-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->dim()) {
    self->stride[dimension] = self->size[dimension+1] * self->stride[dimension+1];
  } else {
    self->stride[dimension] = 1;
  }
  self->size[dimension] = 1;
}

bool THCTensor_isContiguous(THCState *state, const THCTensor *self) {
  int64_t z = 1;
  int d;
  for(d = self->_dim()-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return false;
    }
  }
  return true;
}

bool THCTensor_allContiguous(THCState *state, THCTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_isContiguous(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

ptrdiff_t THCTensor_nElement(THCState *state, const THCTensor *self) {
  if(self->_dim() == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->_dim(); d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THCTensor_retain(THCState *state, THCTensor *self) {
  if(self->flag & TH_TENSOR_REFCOUNTED)
    self->refcount++;
}


void THCTensor_free(THCState *state, THCTensor *self) {
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THCStorage_free(state, self->storage);
      self->refcount.~atomic<int>();
      THFree(self);
    }
  }
}

int THCTensor_getDevice(THCState* state, const THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_getDevice(state, tensor->storage);
}

bool THCTensor_allSameDevice(THCState* state, THCTensor ** inputs, int numInputs) {
  THAssert(numInputs > 0);
  int device = THCTensor_getDevice(state, inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (THCTensor_getDevice(state, inputs[i]) != device) {
      return false;
    }
  }
  return true;
}

bool THCTensor_canUse32BitIndexMath(THCState* state, const THCTensor* t, ptrdiff_t max_elem) {
  ptrdiff_t elements = THCTensor_nElement(state, t);
  if (elements >= max_elem) {
    return false;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = THCTensor_nDimension(state, t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex =
      linearId % THCTensor_size(state, t, i);
    ptrdiff_t curDimOffset = curDimIndex *
      THCTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THCTensor_size(state, t, i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

bool THCTensor_all32BitIndexable(THCState* state, THCTensor** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!THCTensor_canUse32BitIndexMath(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

/* Due to the resize semantics of ops with `out=` keywords, if       */ \
/* the output `tensor` has the same shape as the output of the       */ \
/* reduction operation, then any noncontiguities in the output       */ \
/* `tensor` should be preserved. This needs to be special cased b/c  */ \
/* otherwise, when keepdim=False, the implementations of reduction   */ \
/* ops resize `tensor` to the reduced size with keepdim=True, and    */ \
/* then later squeeze `tensor` to the correct output size, breaking  */ \
/* the contiguity guarantees of the resize semantics.                */ \
void THCTensor_preserveReduceDimSemantics(THCState *state, THCTensor *tensor,
                                          int in_dims, int64_t dimension, int keepdim) {
  int out_dims = THCTensor_nDimension(state, tensor);
  if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {
    THCTensor_unsqueeze1d(state, tensor, tensor, dimension);
  }
}

namespace {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

/*
 A comparator that will sort SizeAndStride structs by stride,
 in ascending order.
 */
int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

}

/* Returns false if there is no possibility that the tensor    */
/* has "overlapping" indices and true otherwise.               */
/* "Overlapping" indices are two+ valid indices that specify   */
/* the same offset within the tensor.                          */
/* The function does this by checking for a sufficient but not */
/* necessary condition of no overlap. In particular, that      */
/* that there exists an ordering of the tensor's dimensions    */
/* that is nicely "nested," with each dimension contained      */
/* within the next one.                                        */
bool THCTensor_maybeOverlappingIndices(THCState* state, const THCTensor* t) {
  /* Extract size/stride arrays; only consider size >1 dims. */
  SizeAndStride info[MAX_CUTORCH_DIMS];

  int dims = THCTensor_nDimension(state, t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = THCTensor_size(state, t, i);

    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride =
        THCTensor_stride(state, t, i);

      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  /* Short-circuits if tensor is a single element.             */
  if (nonSize1Dims == 0) {
    return false;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}

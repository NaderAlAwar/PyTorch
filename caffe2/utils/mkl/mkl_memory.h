#ifndef CAFFE2_UTILS_MKL_MKL_MEMORY_H_
#define CAFFE2_UTILS_MKL_MKL_MEMORY_H_

#include <string>
#include <vector>
#include <mutex>

#include "caffe2/core/tensor.h" // for TIndex
#include "caffe2/core/flags.h" // for TIndex
#include "caffe2/utils/mkl/mkl_dnn_cppwrapper.h"

// A global boolean variable that controls the behavior when we call View() on
// an MKLMemory: if it is set true, then the View() function will actually
// change the underlying storage. If it is set false, an implicit copy is
// triggered but the original storage is not affected.
CAFFE2_DECLARE_bool(caffe2_mkl_implicit_layout_change);

namespace caffe2 {
namespace mkl {

template <typename T>
class PrimitiveWrapper {
 public:
  PrimitiveWrapper() {}
  // Creates a primitive wrapper from an existing primitive. The wrapper
  // takes over ownership.
  explicit PrimitiveWrapper(dnnPrimitive_t primitive) : primitive_(primitive) {}

  template <typename Creator, typename FirstArg, typename... Args>
  PrimitiveWrapper(Creator creator, FirstArg&& arg, Args&&... args) {
    creator(&primitive_, arg, args...);
  }

  ~PrimitiveWrapper() {
    if (primitive_) {
      MKLDNN_CHECK(dnnDelete<T>(primitive_));
    }
  }

  template <typename Creator, typename... Args>
  void Reset(Creator creator, Args&&... args) {
    if (primitive_) {
      MKLDNN_SAFE_CALL(dnnDelete<T>(primitive_));
    }
    creator(&primitive_, args...);
  }

  operator dnnPrimitive_t() const {
    return primitive_;
  }

 private:
  dnnPrimitive_t primitive_ = 0;
  DISABLE_COPY_AND_ASSIGN(PrimitiveWrapper);
};

template <typename T>
class LayoutWrapper {
 public:
  LayoutWrapper() {}
  // Create a user layout from a TensorCPU with the given shapes.
  explicit LayoutWrapper(const TensorCPU& tensor) {
    Reset(tensor);
  }

  // Create an internal layout from the primitive and type.
  LayoutWrapper(const dnnPrimitive_t primitive, const dnnResourceType_t type) {
    Reset(primitive, type);
  }

  // Create a user layout from the given dimension, size and strides.
  LayoutWrapper(
      const size_t dimension,
      const size_t size[],
      const size_t strides[]) {
    Reset(dimension, size, strides);
  }

  // Destructs the layout wrapper.
  ~LayoutWrapper() {
    if (layout_)
      MKLDNN_CHECK(dnnLayoutDelete<T>(layout_));
  }

  // Create a user layout from a TensorCPU with the given shapes.
  void Reset(const TensorCPU& tensor) {
    if (layout_)
      MKLDNN_CHECK(dnnLayoutDelete<T>(layout_));
    CAFFE_ENFORCE(tensor.size(), "Cannot reset with an empty tensor.");
    size_t dimension = tensor.ndim();
    size_t size[dimension];
    size_t strides[dimension];
    for (int i = 0; i < dimension; ++i) {
      size[i] = tensor.dim(dimension - i - 1);
      strides[i] = (i == 0) ? 1 : strides[i - 1] * size[i - 1];
    }
    MKLDNN_SAFE_CALL(dnnLayoutCreate<T>(&layout_, dimension, size, strides));
  }

  // Create an internal layout from the primitive and type.
  void Reset(const dnnPrimitive_t primitive, const dnnResourceType_t type) {
    CAFFE_ENFORCE(primitive, "Cannot reset with an unknwon primitive.");
    CAFFE_ENFORCE(
        type != dnnResourceNumber,
        "Cannot reset with an unknown resource number.");
    if (layout_) {
      MKLDNN_CHECK(dnnLayoutDelete<T>(layout_));
    }
    MKLDNN_SAFE_CALL(
        dnnLayoutCreateFromPrimitive<T>(&layout_, primitive, type));
  }

  // Create a user layout from the given dimension, size and strides.
  void
  Reset(const size_t dimension, const size_t size[], const size_t strides[]) {
    if (layout_)
      MKLDNN_CHECK(dnnLayoutDelete<T>(layout_));
    MKLDNN_SAFE_CALL(dnnLayoutCreate<T>(&layout_, dimension, size, strides));
  }

  operator dnnLayout_t() const {
    return layout_;
  }

 private:
  dnnLayout_t layout_ = 0;
  DISABLE_COPY_AND_ASSIGN(LayoutWrapper);
};

/**
 * @brief A wrapper around an opaque MKL internal resource that has certain
 * layouts and convertion primitives set up.
 *
 * Most of the MKLMemory functions are not thread safe.
 */
template <typename T>
class MKLMemory {
 public:
  // Initializes an empty MKLMemory.
  MKLMemory() {}
  // Initialize an MKLMemory with the given size, strides, dnn
  // primitive and type.
  MKLMemory(
      const size_t dimension,
      const size_t size[],
      const size_t strides[],
      const dnnPrimitive_t primitive = nullptr,
      const dnnResourceType_t type = dnnResourceNumber,
      bool share_mem_if_possible = false) {
    Reset(dimension, size, strides, primitive, type, share_mem_if_possible);
  }

  // Initialize an MKLMemory, with the given dimension assuming a C-contiguous
  // storage.
  template <typename IndexType>
  explicit MKLMemory(
      const vector<IndexType>& dims,
      const dnnPrimitive_t primitive = nullptr,
      const dnnResourceType_t type = dnnResourceNumber,
      bool share_mem_if_possible = false) {
    Reset(dims, primitive, type, share_mem_if_possible);
  }

  // Initialize an MKLMemory with the given size, strides, dnn
  // primitive and type.
  void Reset(
      const size_t dimension,
      const size_t size[],
      const size_t strides[],
      const dnnPrimitive_t primitive = nullptr,
      const dnnResourceType_t type = dnnResourceNumber,
      bool share_mem_if_possible = false) {
    buffer_.reset();
    dims_.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
      dims_[i] = size[dimension - 1 - i];
    }
    user_layout_.Reset(dimension, size, strides);
    if (primitive) {
      layout_.Reset(primitive, type);
    } else {
      layout_.Reset(dimension, size, strides);
    }
    convert_in_.Reset(dnnConversionCreate<T>, user_layout_, layout_);
    convert_out_.Reset(dnnConversionCreate<T>, layout_, user_layout_);
    share_mem_if_possible_ = share_mem_if_possible;
    layout_is_user_layout_ = dnnLayoutCompare<T>(layout_, user_layout_);
    VLOG(2) << "layout is user layout? " << layout_is_user_layout_;
    if (!share_mem_if_possible_) {
      // If we are not going to share memory, we will simply allocate
      // memory upfront.
      buffer();
    }
  }

  // Initialize an MKLMemory, with the given dimension assuming a C-contiguous
  // storage.
  template <typename IndexType>
  void Reset(
      const vector<IndexType>& dims,
      const dnnPrimitive_t primitive = nullptr,
      const dnnResourceType_t type = dnnResourceNumber,
      bool share_mem_if_possible = false) {
    buffer_.reset();
    dims_.resize(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      dims_[i] = dims[i];
    }
    size_t dimension = dims.size();
    size_t size[dimension];
    size_t strides[dimension];
    for (int i = 0; i < dimension; ++i) {
      size[i] = dims[dimension - i - 1];
      strides[i] = (i == 0) ? 1 : strides[i - 1] * size[i - 1];
    }
    user_layout_.Reset(dims.size(), size, strides);
    if (primitive) {
      layout_.Reset(primitive, type);
    } else {
      layout_.Reset(dimension, size, strides);
    }
    convert_in_.Reset(dnnConversionCreate<T>, user_layout_, layout_);
    convert_out_.Reset(dnnConversionCreate<T>, layout_, user_layout_);
    share_mem_if_possible_ = share_mem_if_possible;
    layout_is_user_layout_ = dnnLayoutCompare<T>(layout_, user_layout_);
    VLOG(2) << "layout is user layout? " << layout_is_user_layout_;
    if (!share_mem_if_possible_) {
      // If we are not going to share memory, we will simply allocate
      // memory upfront.
      buffer();
    }
  }

  // Destructs the MKLMemory.
  ~MKLMemory() {}

  void CopyFrom(const void* ptr) {
    if (share_mem_if_possible_ && layout_is_user_layout_) {
      VLOG(2) << "Sharing underlying memory and skip copy.";
      buffer_.reset(const_cast<void*>(ptr), [](void*) -> void {});
    } else {
      VLOG(2) << "Copying external content.";
      MKLDNN_SAFE_CALL(dnnConversionExecute<T>(
          convert_in_, const_cast<void*>(ptr), buffer()));
    }
  }

  void CopyFrom(const TensorCPU& tensor) {
    CAFFE_ENFORCE_EQ(
        tensor.dims(),
        dims_,
        "Dims does not match the expected dims of the resource.");
    CopyFrom(tensor.template data<T>());
  }

  void CopyFrom(const MKLMemory<T>& other) {
    if (share_mem_if_possible_ && dnnLayoutCompare(other.layout_, layout_)) {
      buffer_ = other.buffer_;
    } else {
      PrimitiveWrapper<T> convert(
          dnnConversionCreate<T>, other.layout_, layout_);
      MKLDNN_SAFE_CALL(
          dnnConversionExecute<T>(convert, other.buffer_, buffer()));
    }
  }

  bool ShareFromRaw(const void* ptr) {
    if (share_mem_if_possible_ && layout_is_user_layout_) {
      buffer_.reset(const_cast<void*>(ptr), [](void*) -> void {});
      return true;
    } else {
      return false;
    }
  }

  bool ShareFromTensor(const TensorCPU& tensor) {
    CAFFE_ENFORCE_EQ(
        tensor.dims(),
        dims_,
        "Dims does not match the expected dims of the resource.");
    return ShareFromRaw(tensor.template data<T>());
  }

  bool ShareFrom(const MKLMemory<T>& other) {
    if (share_mem_if_possible_ && dnnLayoutCompare<T>(other.layout_, layout_)) {
      VLOG(2) << "Sharing underlying memory.";
      buffer_ = other.buffer_;
      if (!buffer_.get()) {
        VLOG(2) << "Warning: the source MKLMemory has no content yet, so the "
                   "sharing actually has no effect.";
      }
      return true;
    } else {
      VLOG(2) << "Not sharing underlying memory.";
      return false;
    }
  }

  void CopyTo(void* ptr) const {
    if (buffer_.get() == ptr) {
      // This is already mapping to the same memory region. Skip copy.
      VLOG(2) << "CopyTo does not need actual copying, as we are sharing "
                 "memory with the output.";
      return;
    }
    CAFFE_ENFORCE(
        buffer_.get(), "Canot copy out from an uninitialized MKLMemory.");
    VLOG(2) << "Copy to external memory.";
    MKLDNN_SAFE_CALL(dnnConversionExecute<T>(convert_out_, buffer_.get(), ptr));
  }

  void CopyTo(TensorCPU* tensor) const {
    if (tensor->size() > 0 && buffer_.get() == tensor->mutable_data<T>()) {
      // This is already mapping to the same memory region. Skip copy.
      VLOG(2) << "CopyTo does not need actual copying, as we are sharing "
                 "memory with the output.";
      return;
    }
    tensor->Resize(dims_);
    CopyTo(tensor->mutable_data<T>());
  }

  // Copies to another MKL memory.
  //
  // This function
  void CopyTo(
      MKLMemory<T>* other,
      const dnnPrimitive_t primitive = nullptr,
      const dnnResourceType_t type = dnnResourceNumber) {
    if (buffer_.get() == other->buffer_.get()) {
      VLOG(2) << "CopyTo does not need actual copying, as we are sharing "
                 "memory with the output.";
      // This is already mapping to the same memory region. Skip copy.
      return;
    }
    CAFFE_ENFORCE(
        buffer_.get(), "Canot copy out from an uninitialized MKLMemory.");
    // TODO(jiayq): if primitive creation is a big overhead and we will be
    // consistently copying stuff with fixed src and dst layouts, consider
    // making a cache for the primitive below.
    VLOG(2) << "CopyTo requires copying. Performing direct copy.";
    PrimitiveWrapper<T> convert(
        dnnConversionCreate<T>, layout_, other->layout_);
    if (dnnPrimitive_t(convert) == nullptr ||
        dnnConversionExecute<T>(convert, buffer_.get(), other->buffer()) !=
            E_SUCCESS) {
      VLOG(2) << "Direct copy failed, will need to allocate output.";
      // If CopyTo directly did not succeed, it could be because the target
      // MKLMemory is not having the right layout. In this case we will reset
      // the target and then do another copy.
      other->Reset(dims_, primitive, type);
      PrimitiveWrapper<T> convert2(
          dnnConversionCreate<T>, layout_, other->layout_);
      MKLDNN_SAFE_CALL(
          dnnConversionExecute<T>(convert2, buffer_.get(), other->buffer()));
    }
  }

  inline void* buffer() {
    if (buffer_ == nullptr) {
      CAFFE_ENFORCE(
          layout_ != nullptr, "Trying to allocate buffer but layout is empty.");
      void* allocated = nullptr;
      MKLDNN_SAFE_CALL(dnnAllocateBuffer<T>(&allocated, layout_));
      buffer_.reset(allocated, [](void* ptr) -> void {
        MKLDNN_CHECK(dnnReleaseBuffer<T>(ptr));
      });
    }
    return buffer_.get();
  }

  // MKLDNN does not use const void* even for the inputs, so we will
  // have to use void* and rely on the underlying implementation to make
  // sure that the buffer is actually not changed.
  inline void* buffer() const {
    CAFFE_ENFORCE(
        buffer_ != nullptr, "Trying to refer to an unallocated buffer.");
    return buffer_.get();
  }

  inline const vector<TIndex>& dims() const {
    return dims_;
  }

  inline const int ndim() const { return dims_.size(); }

  inline int dim32(const int i) const {
    CAFFE_ENFORCE_LT(dims_.at(i), std::numeric_limits<int>::max());
    return static_cast<int>(dims_[i]);
  }

  /**
   * Returns the i-th dimension of the tensor. Note that the passed in index
   * must be between 0 (inclusive) and the number of dimensions, otherwise
   * this function will produce a fatal message.
   */
  inline TIndex dim(const int i) const {
    return dims_.at(i);
  }

  inline const LayoutWrapper<T>& layout() const {
    return layout_;
  }

  // Returns a view of the content. We mark this function const, but be noted
  // that the returned std::shared_ptr is not const protected - user discretion
  // is recommended for correctness.
  std::shared_ptr<void> View(
      dnnLayout_t layout_wanted,
      dnnPrimitive_t primitive,
      dnnResourceType_t type) const {
    std::lock_guard<std::mutex> lock(buffer_lock_);
    if (dnnLayoutCompare<T>(layout_wanted, layout_)) {
      // If they are the same, return the original content.
      VLOG(2) << "Creating a view without the need of copying.";
      return std::shared_ptr<void>(buffer_);
    } else {
      void* temp_buffer;
      VLOG(2) << "Creating a view with copying.";
      MKLDNN_SAFE_CALL(dnnAllocateBuffer<T>(&temp_buffer, layout_wanted));
      PrimitiveWrapper<T> convert(
          dnnConversionCreate<T>, layout_, layout_wanted);
      MKLDNN_SAFE_CALL(dnnConversionExecute<T>(
          convert, buffer_.get(), temp_buffer));
      if (FLAGS_caffe2_mkl_implicit_layout_change) {
        VLOG(2) << "Implicit layout change set. "
                   "Changing the underlying storage.";
        // We will need to call Reset to set up all the member variables.
        // This is not thread safe, so we might want to double check if this
        // makes sense in actual use cases.
        const_cast<MKLMemory<T>*>(this)->Reset(
            dims_, primitive, type, share_mem_if_possible_);
        CAFFE_ENFORCE(dnnLayoutCompare<T>(layout_wanted, layout_),
                      "You passed in a target layout that is not "
                      "generated by the given primitive and type.");
        buffer_.reset(temp_buffer, [](void* ptr) -> void {
                MKLDNN_CHECK(dnnReleaseBuffer<T>(ptr));
            });
        return std::shared_ptr<void>(buffer_);
      } else {
        return std::shared_ptr<void>(temp_buffer, [](void* ptr) -> void {
                MKLDNN_CHECK(dnnReleaseBuffer<T>(ptr));
            });
      }
    }
  }

 private:
  bool share_mem_if_possible_;
  bool layout_is_user_layout_;
  // The internal buffer in the specific dnn layout.
  // It is marked mutable but any modification in a const function should
  // be accompanied by the buffer lock, see the View() function.
  mutable std::shared_ptr<void> buffer_;
  // A mutex to control the access of buffer in the View() function.
  mutable std::mutex buffer_lock_;
  // The dimensions in the same order as Caffe2 does. This is used to
  // interface with C2.
  vector<TIndex> dims_;
  // The user dnn layout.
  LayoutWrapper<T> user_layout_;
  // The internal dnn layout.
  LayoutWrapper<T> layout_;
  // The primitive to use to convert from user layout to internal layout
  PrimitiveWrapper<T> convert_in_;
  // The primitive to use to convert from internal layout to user layout
  PrimitiveWrapper<T> convert_out_;

  DISABLE_COPY_AND_ASSIGN(MKLMemory);
};

} // namespace mkl
} // namespace caffe2

#endif // CAFFE2_UTILS_MKL_MKL_MEMORY_H_

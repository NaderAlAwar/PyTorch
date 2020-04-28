#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CuSparseDescriptors.h>

#include <TH/THGeneral.h>

#include <cusparse.h>

#if !defined(_MSC_VER) && defined(__CUDACC__) && CUSPARSE_VERSION >= 10301 // CUDA release >= 10.2 and not windows
#include <library_types.h>
#endif

// LIMITATION (cusparseSpMM): 
// The generic APIs are currently (CUDA 10.1) available for all platforms except Windows. 
// Using these APIs in any other systems will result in compile-time or run-time failures. 
// Their support will be extended in the next releases. 

#if !defined(CUSPARSE_VERSION) || (CUSPARSE_VERSION < 10200)
const char* cusparseGetErrorString(cusparseStatus_t status) {
  switch(status)
  {
    case CUSPARSE_STATUS_SUCCESS:
      return "success";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      return "unknown error";
  }
}
#endif

namespace at { namespace native { namespace sparse { namespace cuda {

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);

  int i_nnz = (int)nnz;
  int i_m = (int)m;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoo2csr(handle, coorowind, i_nnz, i_m, csrrowptr, CUSPARSE_INDEX_BASE_ZERO));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    AT_ERROR("trans must be one of: t, n, c");
  }
}

#if !defined(_MSC_VER) && defined(__CUDACC__) && CUSPARSE_VERSION >= 10301 // CUDA release >= 10.2 and not windows

template<typename T> 
void csrmm2(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda, 
  T *b, int64_t ldb, T beta, T *c, int64_t ldc)
{
  static_assert(std::is_same<float, T>::value || std::is_same<double, T>::value); 
  constexpr auto cusparse_value_type = std::is_same<float, T>::value ? CUDA_R_32F : CUDA_R_64F; 

  if (csrvala == nullptr || b == nullptr || c == nullptr) return; 

  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // cusparseSpMM actually supports int64_t. 
  // In order to support int64 here, index pointers csrrowptra, csrcolinda have to be passed as int64_t. 
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "At the moment, cusparseSpMM only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX, ".", 
    "If you need this, please file an issue on GitHub."
  );

  int64_t ma = m, ka = k; 
  if (transa != 'n') std::swap(ma, ka); 
  auto descA = CuSparseSpMatCsrDescriptor<T, int>(ma, ka, nnz, csrrowptra, csrcolinda, csrvala);

  int64_t kb = k, nb = n;
  if (transb != 'n') std::swap(kb, nb); 
  auto descB = CuSparseDnMatDescriptor<T>(kb, nb, ldb, b);

  auto descC = CuSparseDnMatDescriptor<T>(m, n, ldc, c);

  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize; 
  TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
    handle, opa, opb,     
    &alpha,               
    descA.desc(), descB.desc(), 
    &beta, 
    descC.desc(), 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    &bufferSize           /* output */
  )); 

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(bufferSize);

  TORCH_CUDASPARSE_CHECK(cusparseSpMM(
    handle, opa, opb, 
    &alpha, 
    descA.desc(), descB.desc(), 
    &beta, 
    descC.desc(), 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    dataPtr.get()         /* external buffer */
  )); 
}
template void csrmm2<float>(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, 
  float *b, int64_t ldb, float beta, float *c, int64_t ldc); 
template void csrmm2<double>(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, 
  double *b, int64_t ldb, double beta, double *c, int64_t ldc); 

#else 

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;


  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
  // TODO: Proper fix is to create real descriptor classes
}

// T can only be float or double
template<typename T> 
void csrmm2(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda, 
  T *b, int64_t ldb, T beta, T *c, int64_t ldc)
{
  TORCH_INTERNAL_ASSERT(false, "cusparse csr MM only supports data type of float and double.");
}

template<> void csrmm2<float>(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, 
  float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  Scsrmm2(transa, transb, m, n, k, nnz, alpha, csrvala, csrrowptra, csrcolinda, b, ldb, beta, c, ldc);
} 
template<> void csrmm2<double>(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, 
  double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  Dcsrmm2(transa, transb, m, n, k, nnz, alpha, csrvala, csrrowptra, csrcolinda, b, ldb, beta, c, ldc); 
} 

#endif

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  TORCH_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = at::cuda::getCurrentCUDASparseHandle();
  TORCH_CUDASPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}


}}}} // namespace at::native::sparse::cuda

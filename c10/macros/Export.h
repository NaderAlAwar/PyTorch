#ifndef C10_MACROS_EXPORT_H_
#define C10_MACROS_EXPORT_H_

/* Header file to define the common scaffolding for exported symbols.
 *
 * Export is by itself a quite tricky situation to deal with, and if you are
 * hitting this file, make sure you start with the background here:
 * - Linux: https://gcc.gnu.org/wiki/Visibility
 * - Windows:
 * https://docs.microsoft.com/en-us/cpp/cpp/dllexport-dllimport?view=vs-2017
 *
 * Do NOT include this file directly. Instead, use c10/macros/Macros.h
 */

// You do not need to edit this part of file unless you are changing the core
// pytorch export abstractions.
//
// This part defines the C10 core export and import macros. This is controlled
// by whether we are building shared libraries or not, which is determined
// during build time and codified in c10/core/cmake_macros.h.
// When the library is built as a shared lib, EXPORT and IMPORT will contain
// visibility attributes. If it is being built as a static lib, then EXPORT
// and IMPORT basically have no effect.

// As a rule of thumb, you should almost NEVER mix static and shared builds for
// libraries that depend on c10. AKA, if c10 is built as a static library, we
// recommend everything dependent on c10 to be built statically. If c10 is built
// as a shared library, everything dependent on it should be built as shared. In
// the PyTorch project, all native libraries shall use the macro
// C10_BUILD_SHARED_LIB to check whether pytorch is building shared or static
// libraries.

#ifdef _WIN32
#if defined(C10_BUILD_SHARED_LIBS)
#define C10_EXPORT __declspec(dllexport)
#define C10_IMPORT __declspec(dllimport)
#else
#define C10_EXPORT
#define C10_IMPORT
#endif
#define C10_EXPORT_ALIGNED(n) C10_EXPORT alignas(n)
#define C10_IMPORT_ALIGNED(n) C10_IMPORT alignas(n)
#else // _WIN32
#if defined(__GNUC__)
#define C10_EXPORT __attribute__((__visibility__("default")))
#define C10_EXPORT_ALIGNED(n) __attribute__((__visibility__("default"), aligned(n)))
#else // defined(__GNUC__)
#define C10_EXPORT
#define C10_EXPORT_ALIGNED(n) alignas(n)
#endif // defined(__GNUC__)
#define C10_IMPORT C10_EXPORT
#define C10_IMPORT_ALIGNED(n) C10_EXPORT_ALIGNED(n)
#endif // _WIN32

// Definition of an adaptive XX_API macro, that depends on whether you are
// building the library itself or not, routes to XX_EXPORT and XX_IMPORT.
// Basically, you will need to do this for each shared library that you are
// building, and the instruction is as follows: assuming that you are building
// a library called libawesome.so. You should:
// (1) for your cmake target (usually done by "add_library(awesome, ...)"),
//     define a macro called AWESOME_BUILD_MAIN_LIB using
//     target_compile_options.
// (2) define the AWESOME_API macro similar to the one below.
// And in the source file of your awesome library, use AWESOME_API to
// annotate public symbols.

// Here, for the C10 library, we will define the macro C10_API for both import
// and export.

// This one is being used by libc10.so
#ifdef C10_BUILD_MAIN_LIB
#define C10_API C10_EXPORT
#define C10_API_ALIGNED(n) C10_EXPORT_ALIGNED(n)
#else
#define C10_API C10_IMPORT
#define C10_API_ALIGNED(n) C10_IMPORT_ALIGNED(n)
#endif

// This one is being used by libcaffe2.so
#ifdef CAFFE2_BUILD_MAIN_LIB
#define CAFFE2_API C10_EXPORT
#else
#define CAFFE2_API C10_IMPORT
#endif

#endif // C10_MACROS_MACROS_H_

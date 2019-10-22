#pragma once

#include <exception>
#include <string>
#include <memory>
#include <queue>
#include <mutex>

#include <torch/csrc/THP_export.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include "c10/util/StringUtil.h"
#include "c10/util/Exception.h"

/// NOTE [ Conversion Cpp Python Warning ]
/// The warning handler cannot set python warnings immediately
/// as it requires acquirering the GIL (potential deadlock)
/// and would need to properly exit if the warning raised a
/// python error. To solve this, we buffer the warnings and
/// process them when we go back to python.
/// This leads to the following cases:
/// - The GIL is acquired in the PyWarningHandler destructor
///   - If there is no Error raised in the inner try/catch, the
///     bufferred warnings are processed as python warnings.
///     - If they don't raise an error, the function process with the
///       original return code.
///     - If any of them raise an error, the error code is set and
///       the destructor will raise a python_error() that will be
///       caught by the outer try/catch that will be able to change
///       the return value of the function to reflect the error.
///   - If an Error was raised in the inner try/catch, the inner try/catch
///     must set the python error. The buffered warnings are then
///     processed as cpp warnings as we cannot predict before hand
///     whether a python warning will raise an error or not and we
///     cannot handle two errors at the same time.
#define HANDLE_TH_ERRORS                                           \
  try {                                                            \
    torch::PyWarningHandler __enforce_warning_buffer;          \
    try{

#define CATCH_TH_ERRORS(retval)                                      \
    catch (python_error & e) {                                       \
      return retval;                                                 \
    }                                                                \
    catch (const c10::IndexError& e) {                               \
      auto msg = torch::processErrorMsg(e.what_without_backtrace()); \
      PyErr_SetString(PyExc_IndexError, msg.c_str());                \
      return retval;                                                 \
    }                                                                \
    catch (const c10::Error& e) {                                    \
      auto msg = torch::processErrorMsg(e.what_without_backtrace()); \
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());              \
      return retval;                                                 \
    }                                                                \
    catch (torch::PyTorchError & e) {                                \
      auto msg = torch::processErrorMsg(e.what());                   \
      PyErr_SetString(e.python_type(), msg.c_str());                 \
      return retval;                                                 \
    }                                                                \
    catch (const std::exception& e) {                                \
      auto msg = torch::processErrorMsg(e.what());                   \
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());              \
      return retval;                                                 \
    }

#define END_HANDLE_TH_ERRORS_RET(retval)                             \
    }                                                                \
    CATCH_TH_ERRORS(retval)                                          \
  }                                                                  \
  CATCH_TH_ERRORS(retval)

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(nullptr)

extern PyObject *THPException_FatalError;

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
struct python_error : public std::exception {
  python_error() : type(nullptr), value(nullptr), traceback(nullptr) {}

  python_error(const python_error &other) : type(other.type), value(other.value), traceback(other.traceback) {
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
  }

  python_error(python_error&& other) {
    type = other.type;
    value = other.value;
    traceback = other.traceback;
    other.type = nullptr;
    other.value = nullptr;
    other.traceback = nullptr;
  }

  ~python_error() override {
    if (type || value || traceback) {
      AutoGIL gil;
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  /** Saves the exception so that it can be re-thrown on a different thread */
  inline void persist() {
    if (type) return; // Don't overwrite exceptions
    // PyErr_Fetch overwrites the pointers
    AutoGIL gil;
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Fetch(&type, &value, &traceback);
  }

  /** Sets the current Python error from this exception */
  inline void restore() {
    if (!type) return;
    // PyErr_Restore steals references
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type, value, traceback);
  }

  PyObject* type;
  PyObject* value;
  PyObject* traceback;
};

#ifdef _THP_CORE

bool THPException_init(PyObject *module);
#endif

namespace torch {

THP_CLASS std::string processErrorMsg(std::string str);

// Abstract base class for exceptions which translate to specific Python types
struct PyTorchError : public std::exception {
  virtual PyObject* python_type() = 0;
  const char* what() const noexcept override {
    return msg.c_str();
  }
  std::string msg;
};

// Translates to Python IndexError
struct IndexError : public PyTorchError {
  IndexError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_IndexError;
  }
};

// Translates to Python TypeError
struct TypeError : public PyTorchError {
  TORCH_API TypeError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_TypeError;
  }
};

// Translates to Python ValueError
struct ValueError : public PyTorchError {
  ValueError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_ValueError;
  }
};

// ATen warning handler for Python
struct PyWarningHandler: at::WarningHandler {
public:
/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
  PyWarningHandler() noexcept(true);
  ~PyWarningHandler() noexcept(false);

  void process(const at::SourceLocation &source_location,
               const std::string &msg) override;

void mark_overlapping();

private:
  using warning_buffer_t =
    std::vector<std::pair<c10::SourceLocation, std::string>>;

  warning_buffer_t warning_buffer_;
  // To avoid deadlocks, if both the GIL and this lock is needed,
  // The GIL needs to be acquired first.
  std::mutex warning_buffer_mutex_;
  at::WarningHandler* prev_handler_;

  // Since our warning handler is global, we want to notify the user if
  // there is a risk of the warning being raised by another python thread
  // than the one that caused the warning.
  bool overlapping_;
  static constexpr char* PYWARNING_MAYBE_INVALID_PYTHON_STACKTRACE =
    "The following warnings happened in a multithreaded or nested setting and so \
the python stack traces below might be incorrect.";
};

} // namespace torch

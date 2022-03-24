#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace c10 {

// `SymInt` is a C++ wrapper class around int64_t data_ which  and is used to
// represent concrete dimension values. 
//
// `SymInt` is also a data type in Pytorch that can be used in function schemas
// to enable tracing.
//
// `SymInt` is introduced to enable tracing arithmetic 
// operations on symbolic integers (e.g. sizes). Tracing symbolic sizes will
// allow LTC and AOTAutograd representing dynamic shapes in expression graphs
// faithfully without baking in concrete dimension values.
//
// To trace the operations, SymInt will overload arithmetic operators (e.g. +, -, *)
// and will provide overloads taking SymInt for commonly used math functions.
//
// SymInt will be extenteded to represent a union structure Union[int64_t, SymbolicIntNode*]
// which will be implemented as a single packed int64_t field named data_.
//
// data_ can be either a plain int64_t or (1 << 63 | `index`). `index` points to
// SymbolicIntNode* that will be responsible for constructing an IR node for
// a traced operation to represent it in LTC or Fx graphs.
class TORCH_API SymInt {
    public:
        SymInt(int64_t d):
        data_(d) {};

        int64_t expect_int const() {
            // we are dealing with concrete ints only for now
            return data_;
        }

        bool is_symbolic const() {
            return false;
        }

        bool operator==(const SymInt& p2) const
        {
            TORCH_INTERNAL_ASSERT("NYI");
            return false;
        }

        SymInt operator+(SymInt sci) const {
            return data_ + sci.data_;
        }

    private:
        int64_t data_;
};

TORCH_API std::ostream& operator<<(std::ostream& os, SymInt s);
}

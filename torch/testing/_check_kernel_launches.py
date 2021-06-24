import os
import re
import regex
import sys

__all__ = [
    "check_code_for_cuda_kernel_launches",
    "check_cuda_kernel_launches",
]


# Regular expression identifies a kernel launch indicator by
# finding something approximating the pattern
# `<<<launch_args>>>(arguments);`
# It then requires that `C10_CUDA_KERNEL_LAUNCH_CHECK` be
# the next command.
#
# Some characters are allowed between the end of the kernel
# launch and the check. At the moment, this includes everything
# except `;` and `}`. A `;` would indicate an intervening
# statement and a `}` would indicate leaving a clause.
# In actuality, only whitespace, line numbers, and `\` for
# continuing macros should be present.
#
# This should be sufficient to detect and fix most problem
# instances and can be refined.
kernel_launch_regex = regex.compile(r"""(
    ^.*<<<[^>]+>>>         # Identifies kernel launch. We use `^.*` to get more context for fixing the issue
    \s*                    # Maybe some whitespace (includes newlines)
    (\(([^)(]+|(?2))*+\)); # Recursive regex that finds matching end paren followed by semi-colon
    (?!                    # Negative lookahead: we trigger if we don't find the launch guard
        [^;}]*             # Match any character that wouldn't indicate we've ended a statement/clause
        C10_CUDA_KERNEL_LAUNCH_CHECK\(\);  # Kernel launch guard!
    )             # End negative lookahead
)""", flags=re.MULTILINE | re.VERBOSE)


def check_code_for_cuda_kernel_launches(code, filename=None):
    """Checks code for CUDA kernel launches without cuda error checks.

    Args:
        filename - Filename of file containing the code. Used only for display
                   purposes, so you can put anything here.
        code     - The code to check

    Returns:
        The number of unsafe kernel launches in the code
    """
    if filename is None:
        filename = "##Python Function Call##"

    # We break the code apart and put it back together to add
    # helpful line numberings for identifying problem areas
    code = enumerate(code.split("\n"))                             # Split by line breaks
    code = [f"{lineno}: {linecode}" for lineno, linecode in code]  # Number the lines
    code = '\n'.join(code)                                         # Put it back together

    results = kernel_launch_regex.findall(code)                    # Search for bad launches
    for r in results:
        print(f"Missing C10_CUDA_KERNEL_LAUNCH_CHECK in '{filename}'. Context:\n{r[0]}", file=sys.stderr)
    return len(results)


def check_file(filename):
    """Checks a file for CUDA kernel launches without cuda error checks

    Args:
        filename - File to check

    Returns:
        The number of unsafe kernel launches in the file
    """
    if not (filename.endswith(".cu") or filename.endswith(".cuh")):
        return 0
    fo = open(filename, "r")
    contents = fo.read()
    unsafeCount = check_code_for_cuda_kernel_launches(contents, filename)
    fo.close()
    return unsafeCount


def check_cuda_kernel_launches():
    """Checks all pytorch code for CUDA kernel launches without cuda error checks

    Returns:
        The number of unsafe kernel launches in the codebase
    """
    torch_dir = os.path.dirname(os.path.realpath(__file__))
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent torch
    torch_dir = os.path.dirname(torch_dir)  # Go up to parent caffe2

    kernels_without_checks = 0
    files_without_checks = []
    for root, dirnames, filenames in os.walk(torch_dir):
        # `$BASE/build` and `$BASE/torch/include` are generated
        # so we don't want to flag their contents
        if root == os.path.join(torch_dir, "build") or root == os.path.join(torch_dir, "torch/include"):
            # Curtail search by modifying dirnames and filenames in place
            # Yes, this is the way to do this, see `help(os.walk)`
            dirnames[:] = []
            continue

        for x in filenames:
            filename = os.path.join(root, x)
            file_result = check_file(filename)
            if file_result > 0:
                kernels_without_checks += file_result
                files_without_checks.append(filename)

    if kernels_without_checks > 0:
        count_str = f"Found {kernels_without_checks} instances in " \
                    f"{len(files_without_checks)} files where kernel " \
                    "launches didn't have checks."
        print(count_str, file=sys.stderr)
        print("Files without checks:", file=sys.stderr)
        for x in files_without_checks:
            print(f"\t{x}", file=sys.stderr)
        print(count_str, file=sys.stderr)

    return kernels_without_checks


if __name__ == "__main__":
    unsafe_launches = check_cuda_kernel_launches()
    sys.exit(0)

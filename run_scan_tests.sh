#!/bin/bash

pytest test/export/test_export.py::TestExport::test_export_associative_scan_symbol_dim test/export/test_export.py::TestExport::test_export_associative_scan_symbol_scandim test/export/test_export.py::TestExport::test_export_associative_scan_lifted_buffers test/inductor/test_op_dtype_prop.py::TestCaseCUDA::test_assoc_scan_cuda test/inductor/test_control_flow.py::AssociativeScanTests test/functorch/test_control_flow.py::AssociativeScanTests test/functorch/test_control_flow.py::TestControlFlow::test_scan_associative_scan

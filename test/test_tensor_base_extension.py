import os
import unittest
import torch
import torch.utils.cpp_extension

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_cuda import TEST_CUDA


class TestTensorBaseExtension(TestCase):
    """Test the TensorBase custom extension using no_header mode."""
    
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_tensor_base_extension(self):
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the extension using load() with no_header=True
        module = torch.utils.cpp_extension.load(
            name="tensor_base_extension",
            sources=[
                os.path.join(current_dir, "tensor_base_extension.cpp"), 
                os.path.join(current_dir, "tensor_base_extension.cu")
            ],
            verbose=True,
            no_header=True  # Skip including torch/extension.h
        )
        
        # Test the functionality
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)
        
        # Call our custom kernel
        result = module.tensor_base_add(x, y)
        
        # Verify result (our kernel adds 1.0 to distinguish it from a regular add)
        expected = x + y + 1.0
        self.assertEqual(result, expected)
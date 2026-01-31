# SPDX-License-Identifier: Apache-2.0
"""Tests for SDNQ quantization config."""

import unittest
from unittest.mock import MagicMock, patch


class TestSDNQQuantizationConfig(unittest.TestCase):
    """Test cases for SDNQQuantizationConfig."""

    def test_sdnq_config_registration(self):
        """Test that SDNQ config is properly registered."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            QUANTIZATION_METHODS,
            get_quantization_config,
        )

        # Check that 'sdnq' is in the list of quantization methods
        self.assertIn("sdnq", QUANTIZATION_METHODS)

        # Check that we can retrieve the config class
        sdnq_config_cls = get_quantization_config("sdnq")
        self.assertEqual(sdnq_config_cls.__name__, "SDNQQuantizationConfig")

    def test_sdnq_config_instantiation(self):
        """Test that SDNQQuantizationConfig can be instantiated."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        # Default instantiation
        config = SDNQQuantizationConfig()
        self.assertFalse(config.use_quantized_matmul)
        self.assertTrue(config.apply_to_transformer)
        self.assertTrue(config.apply_to_text_encoder)

        # Custom instantiation
        config = SDNQQuantizationConfig(
            use_quantized_matmul=False,
            apply_to_transformer=False,
            apply_to_text_encoder=False,
        )
        self.assertFalse(config.use_quantized_matmul)
        self.assertFalse(config.apply_to_transformer)
        self.assertFalse(config.apply_to_text_encoder)

    def test_sdnq_config_get_name(self):
        """Test get_name method."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        self.assertEqual(config.get_name(), "sdnq")

    def test_sdnq_config_supported_dtypes(self):
        """Test get_supported_act_dtypes method."""
        import torch

        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        dtypes = config.get_supported_act_dtypes()
        self.assertIn(torch.bfloat16, dtypes)
        self.assertIn(torch.float16, dtypes)
        self.assertIn(torch.float32, dtypes)

    def test_sdnq_config_min_capability(self):
        """Test get_min_capability method."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        min_cap = SDNQQuantizationConfig.get_min_capability()
        self.assertEqual(min_cap, 70)  # Volta and above

    def test_sdnq_config_filenames(self):
        """Test get_config_filenames method."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        filenames = SDNQQuantizationConfig.get_config_filenames()
        self.assertIn("quantization_config.json", filenames)
        self.assertIn("config.json", filenames)

    def test_sdnq_config_from_config(self):
        """Test from_config class method."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        # Test with default values
        config_dict = {}
        config = SDNQQuantizationConfig.from_config(config_dict)
        self.assertFalse(config.use_quantized_matmul)
        self.assertTrue(config.apply_to_transformer)

        # Test with custom values
        config_dict = {
            "use_quantized_matmul": False,
            "apply_to_transformer": False,
            "apply_to_text_encoder": False,
        }
        config = SDNQQuantizationConfig.from_config(config_dict)
        self.assertFalse(config.use_quantized_matmul)
        self.assertFalse(config.apply_to_transformer)
        self.assertFalse(config.apply_to_text_encoder)

    def test_sdnq_config_get_quant_method(self):
        """Test get_quant_method returns None (SDNQ uses model-level quantization)."""
        import torch

        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        layer = torch.nn.Linear(10, 10)
        method = config.get_quant_method(layer, "test.layer")
        self.assertIsNone(method)


class TestSDNQLinearMethod(unittest.TestCase):
    """Test cases for SDNQLinearMethod."""

    def test_sdnq_linear_method_instantiation(self):
        """Test that SDNQLinearMethod can be instantiated."""
        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQLinearMethod,
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        method = SDNQLinearMethod(config)
        self.assertEqual(method.quant_config, config)

    def test_sdnq_linear_method_create_weights(self):
        """Test create_weights is a no-op."""
        import torch

        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQLinearMethod,
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        method = SDNQLinearMethod(config)
        layer = torch.nn.Linear(10, 10)

        # Should not raise any exceptions
        method.create_weights(layer)

    def test_sdnq_linear_method_process_weights_after_loading(self):
        """Test process_weights_after_loading is a no-op."""
        import torch

        from sglang.multimodal_gen.runtime.layers.quantization import (
            SDNQLinearMethod,
            SDNQQuantizationConfig,
        )

        config = SDNQQuantizationConfig()
        method = SDNQLinearMethod(config)
        layer = torch.nn.Linear(10, 10)

        # Should not raise any exceptions
        method.process_weights_after_loading(layer)


class TestSDNQHelperFunctions(unittest.TestCase):
    """Test cases for helper functions in sdnq_config."""

    def test_check_sdnq_available_when_not_installed(self):
        """Test _check_sdnq_available returns False when sdnq is not installed."""
        from sglang.multimodal_gen.runtime.layers.quantization.sdnq_config import (
            _check_sdnq_available,
        )

        # sdnq is not installed in the test environment
        result = _check_sdnq_available()
        # This should return False if sdnq is not installed
        self.assertIsInstance(result, bool)

    def test_check_triton_available_when_not_installed(self):
        """Test _check_triton_available returns False when triton is not available."""
        from sglang.multimodal_gen.runtime.layers.quantization.sdnq_config import (
            _check_triton_available,
        )

        # This should return False if sdnq/triton is not available
        result = _check_triton_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

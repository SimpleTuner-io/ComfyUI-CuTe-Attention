import concurrent.futures
import threading
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from cute_attention import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.original = F.scaled_dot_product_attention
        self.process_mode = runtime._process_mode
        runtime._process_mode = None
        F.scaled_dot_product_attention = runtime._native_sdpa
        self.addCleanup(setattr, F, "scaled_dot_product_attention", self.original)
        self.addCleanup(setattr, runtime, "_process_mode", self.process_mode)
        self.q = torch.randn(1, 2, 5, 8)

    def test_process_override_can_be_disabled_without_recursion(self):
        expected = runtime._native_sdpa(self.q, self.q, self.q)
        backend = Mock(return_value=expected)
        with patch.dict(runtime._BACKENDS, auto=backend):
            runtime.set_process_mode("auto")
            actual = F.scaled_dot_product_attention(self.q, self.q, self.q)
            self.assertIs(actual, expected)
            backend.assert_called_once()
            runtime.set_process_mode(None)
            torch.testing.assert_close(F.scaled_dot_product_attention(self.q, self.q, self.q), expected)
            self.assertEqual(backend.call_count, 1)

    def test_model_scope_restores_after_exception_and_nested_scope(self):
        with patch.dict(
            runtime._BACKENDS, auto=Mock(return_value="auto"), strict=Mock(return_value="strict")
        ):
            runtime.set_process_mode("auto")
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                with runtime.model_mode("strict"):
                    self.assertEqual(F.scaled_dot_product_attention(self.q, self.q, self.q), "strict")
                    with runtime.model_mode("auto"):
                        self.assertEqual(F.scaled_dot_product_attention(self.q, self.q, self.q), "auto")
                    self.assertEqual(F.scaled_dot_product_attention(self.q, self.q, self.q), "strict")
                    raise RuntimeError("model failure")
            self.assertEqual(F.scaled_dot_product_attention(self.q, self.q, self.q), "auto")

    def test_model_scope_does_not_leak_to_other_thread(self):
        barrier = threading.Barrier(2)

        def run_scoped():
            with runtime.model_mode("strict"):
                barrier.wait(timeout=10)
                barrier.wait(timeout=10)
                return F.scaled_dot_product_attention(self.q, self.q, self.q)

        with patch.dict(
            runtime._BACKENDS, auto=Mock(return_value="auto"), strict=Mock(return_value="strict")
        ):
            runtime.set_process_mode("auto")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                scoped = executor.submit(run_scoped)
                barrier.wait(timeout=10)
                other = F.scaled_dot_product_attention(self.q, self.q, self.q)
                barrier.wait(timeout=10)
                self.assertEqual(scoped.result(timeout=10), "strict")
                self.assertEqual(other, "auto")

    def test_auto_native_fallback_preserves_arguments_and_gradients(self):
        runtime.set_process_mode("auto")
        for causal, scale, gqa in (
            (False, None, False),
            (True, 0.3, False),
            (False, -0.3, False),
            (False, 0.0, True),
        ):
            with self.subTest(causal=causal, scale=scale, gqa=gqa):
                q = torch.randn(2, 4, 3, 8, requires_grad=True)
                k = torch.randn(2, 2 if gqa else 4, 7, 8, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                kwargs = dict(is_causal=causal, scale=scale, enable_gqa=gqa)
                expected = runtime._native_sdpa(q, k, v, **kwargs)
                actual = F.scaled_dot_product_attention(q, k, v, **kwargs)
                torch.testing.assert_close(actual, expected)
                wanted = torch.autograd.grad(expected.square().sum(), (q, k, v))
                got = torch.autograd.grad(actual.square().sum(), (q, k, v))
                for a, b in zip(got, wanted):
                    torch.testing.assert_close(a, b)

    def test_conflicting_patch_is_preserved(self):
        other_extension = Mock()
        F.scaled_dot_product_attention = other_extension
        with self.assertRaisesRegex(RuntimeError, "Another extension"):
            runtime.set_process_mode("auto")
        self.assertIs(F.scaled_dot_product_attention, other_extension)
        self.assertIsNone(runtime._process_mode)

    def test_selected_backend_errors_propagate(self):
        backend = Mock(side_effect=RuntimeError("kernel failure"))
        with patch.dict(runtime._BACKENDS, auto=backend):
            runtime.set_process_mode("auto")
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                F.scaled_dot_product_attention(self.q, self.q, self.q)

    def test_strict_rejects_cpu(self):
        runtime.set_process_mode("strict")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            F.scaled_dot_product_attention(self.q, self.q, self.q)


if __name__ == "__main__":
    unittest.main()

"""Live deterministic-policy checks at atomic backward execution."""

import unittest
import warnings
from contextlib import contextmanager

import torch

from cute_attention.vendor.kohakufa_cute.determinism import check_atomic_backward


@contextmanager
def deterministic(enabled, *, warn_only=False):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


class DeterminismPolicyTests(unittest.TestCase):
    def test_policy_changes_are_live_and_warning_only_continues(self):
        with deterministic(False), warnings.catch_warnings(record=True) as caught:
            check_atomic_backward()
        self.assertFalse(caught)
        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            check_atomic_backward()
        with deterministic(True, warn_only=True), self.assertWarnsRegex(UserWarning, "atomic_backward"):
            check_atomic_backward()
        with deterministic(False):
            check_atomic_backward()

    def test_python_warp_guard_precedes_optional_runtime_and_tensor_access(self):
        from cute_attention.vendor.kohakufa_cute.fa4_sm80._attention import _backward

        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            _backward(*([None] * 11))

    def test_python_hopper_guard_precedes_padding_and_optional_runtime(self):
        from cute_attention.vendor.kohakufa_cute.fa4_hopper._attention import _Attention

        with deterministic(True), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
            _Attention.backward(None, None)

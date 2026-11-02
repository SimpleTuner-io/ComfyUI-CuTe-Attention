"""Automatic training eligibility, live metadata and actual API routing."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cute_attention.vendor.kohakufa_cute.fa4_sm80 import rtx4090


class AutomaticPolicyTests(unittest.TestCase):
    def tearDown(self):
        rtx4090._automatic_metadata_predicate.cache_clear()

    def test_only_function_handle_is_cached(self):
        rtx4090._automatic_metadata_predicate.cache_clear()
        function = Mock()
        loader = Mock(return_value=SimpleNamespace(rtx4090_automatic_metadata_compatible=function))
        with patch.dict(
            sys.modules, {"cute_attention.vendor.kohakufa_cute.native": SimpleNamespace(extension=loader)}
        ):
            self.assertIs(rtx4090._automatic_metadata_predicate(), function)
            self.assertIs(rtx4090._automatic_metadata_predicate(), function)
        loader.assert_called_once()

    def test_unsupported_policy_rejects_without_loading(self):
        for options in (
            (None, 0.1, None),
            (object(), 0, None),
            (None, 0, -0.1),
            (None, 0, 0),
            (None, 0, float("nan")),
        ):
            with patch.object(rtx4090, "_automatic_metadata_predicate", side_effect=AssertionError("loaded")):
                self.assertFalse(
                    rtx4090.automatic_compatible(
                        None, None, None, options[0], options[1], False, options[2], False
                    )
                )

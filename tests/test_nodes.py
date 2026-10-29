import copy
import importlib
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch.nn.functional as F

from cute_attention import runtime

# Node policy can be checked without importing ComfyUI's GPU/model stack.
# The GPU integration suite separately uses the real ModelPatcher and wrappers.
extension_stub = types.ModuleType("comfy.patcher_extension")
extension_stub.WrappersMP = types.SimpleNamespace(APPLY_MODEL="apply_model")
with patch.dict(sys.modules, {"comfy": types.ModuleType("comfy"), "comfy.patcher_extension": extension_stub}):
    CuTeAttention = importlib.import_module("cute_attention.nodes").CuTeAttention


class Model:
    def __init__(self):
        self.model_options = {"transformer_options": {"other_patch": "preserved"}}
        self.wrappers = {}

    def clone(self):
        return copy.deepcopy(self)

    def set_model_optimized_attention(self, function):
        self.model_options["transformer_options"]["optimized_attention_override"] = function

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.get(kind, {}).pop(key, None)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.original = F.scaled_dot_product_attention
        self.process_mode = runtime._process_mode
        F.scaled_dot_product_attention = runtime._native_sdpa
        runtime._process_mode = None
        self.addCleanup(setattr, F, "scaled_dot_product_attention", self.original)
        self.addCleanup(setattr, runtime, "_process_mode", self.process_mode)

    def test_clone_preserves_input_and_replaces_own_wrapper(self):
        source = Model()
        patched = CuTeAttention().patch(source, scope="model")[0]
        twice = CuTeAttention().patch(patched, scope="model", mode="strict")[0]
        self.assertNotIn("optimized_attention_override", source.model_options["transformer_options"])
        self.assertEqual(source.wrappers, {})
        self.assertEqual(twice.model_options["transformer_options"]["other_patch"], "preserved")
        self.assertEqual(len(twice.wrappers["apply_model"]["cute_attention"]), 1)
        self.assertIsNone(runtime._process_mode)

    def test_default_process_scope_and_explicit_disable(self):
        source = Model()
        CuTeAttention().patch(source)
        self.assertEqual(runtime._process_mode, "auto")
        disabled = CuTeAttention().patch(source, enabled=False)[0]
        self.assertIs(disabled, source)
        self.assertIsNone(runtime._process_mode)
        self.assertTrue(math.isnan(CuTeAttention.IS_CHANGED()))

    def test_model_scope_keeps_existing_process_setting(self):
        runtime.set_process_mode("strict")
        CuTeAttention().patch(Model(), scope="model")
        self.assertEqual(runtime._process_mode, "strict")


if __name__ == "__main__":
    unittest.main()

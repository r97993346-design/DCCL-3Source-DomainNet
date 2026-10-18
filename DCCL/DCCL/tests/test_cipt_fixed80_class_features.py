"""Small CPU-only checks for frozen 80-template class prototypes.

Run: python -m unittest discover -s tests -p 'test_cipt_fixed80_class_features.py'
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.algorithms.cipt_modules import CausalDecomposition
from domainbed.algorithms.cipt_prompt import CIPTTextFeatures


def toy_features(mode):
    # Bypass the full pretrained CLIP constructor for a quick unit test.
    features = CIPTTextFeatures.__new__(CIPTTextFeatures)
    nn.Module.__init__(features)
    features.class_text_mode = mode
    features.register_buffer(
        "b5b_text_bank",
        torch.tensor([
            [[1., 0., 0.], [0., 1., 0.]],
            [[0., 0., 1.], [0., 0., 1.]],
        ]),
    )
    return features


class Fixed80ClassFeaturesTests(unittest.TestCase):
    def test_ensemble_means_and_normalizes_per_class(self):
        model = toy_features("ensemble80")
        actual = model.class_features()
        expected = F.normalize(
            model.b5b_text_bank.mean(dim=1), dim=-1
        )
        self.assertEqual(tuple(actual.shape), (2, 3))
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(actual.requires_grad)

    def test_es_adapters_still_receive_gradients(self):
        text = toy_features("ensemble80").class_features()
        decomposition = CausalDecomposition(3)
        visual = F.normalize(torch.tensor([
            [0.6, 0.5, 0.1],
            [0.1, 0.1, 0.9],
            [0.5, 0.4, 0.2],
            [0.2, 0.2, 0.8],
        ]), dim=-1)
        labels = torch.tensor([0, 1, 0, 1])
        e, s = decomposition(visual)
        e_logits = 10 * F.normalize(e, dim=-1) @ text.T
        s_logits = 10 * F.normalize(s, dim=-1) @ text.T
        # CE_E and KL(U || p_S), up to uniform entropy constant.
        loss = F.cross_entropy(e_logits, labels) - F.log_softmax(
            s_logits, dim=-1
        ).mean()
        loss.backward()
        self.assertIsNotNone(decomposition.causal_adapter.weight.grad)
        self.assertIsNotNone(decomposition.spurious_adapter.weight.grad)
        self.assertGreater(
            decomposition.causal_adapter.weight.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            decomposition.spurious_adapter.weight.grad.abs().sum().item(), 0
        )

    def test_invalid_class_mode_fails(self):
        model = toy_features("typo")
        with self.assertRaises(ValueError):
            model.class_features()


if __name__ == "__main__":
    unittest.main()

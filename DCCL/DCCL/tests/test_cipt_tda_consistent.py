"""CPU functional regressions for candidate-wise TDA.

To run from DCCL/DCCL:
  python -m unittest discover -s tests -p 'test_cipt_tda_consistent.py' -v
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "CIPT"))  # upstream source
import torch
from torch import nn
import torch.nn.functional as F
from cipt.losses import cipt_loss
from cipt_tda_consistent import ConsistentCIPTModel


class ToyAugmentation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mapping = nn.Linear(dim, dim)

    def forward(self, e, p):
        return self.mapping(e[:, None, :] + p)


class ToyCIPT(ConsistentCIPTModel):
    """Minimal deterministic module for fast gradient/shape tests on CPU."""

    def __init__(self):
        nn.Module.__init__(self)
        dim, classes, templates = 5, 3, 4
        self.register_buffer("diverse_text_features", torch.randn(classes, templates, dim))
        self.sample_templates = False
        self.class_conditioned_templates = True
        self.num_diverse_templates = 2
        self.class_chunk_size = 1
        self.causal_adapter = nn.Linear(dim, dim)
        self.spurious_adapter = nn.Linear(dim, dim)
        self.diversity_augmentation = ToyAugmentation(dim)
        self.text_param = nn.Parameter(torch.randn(classes, dim))
        self.clip_model = nn.Module()
        self.text_encoder = nn.Identity()  # upstream train() keeps encoder in eval mode
        self.clip_model.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image_features(self, images):
        return F.normalize(images.float(), dim=-1)

    def encode_prompt_features(self):
        return F.normalize(self.text_param, dim=-1)


class ConsistentTDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.model = ToyCIPT()
        self.images = torch.randn(4, 5)
        self.labels = torch.tensor([0, 1, 2, 0])

    def test_gold_labels_do_not_enter_forward_logits(self):
        self.model.eval()
        a = self.model(self.images, labels=self.labels).interventional_logits
        b = self.model(self.images, labels=self.labels.roll(1)).interventional_logits
        self.assertEqual(tuple(a.shape), (4, 2, 3))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_candidate_scoring_equals_explicit_reference(self):
        self.model.eval()
        e = self.model.causal_adapter(self.model.encode_image_features(self.images))
        text = self.model.encode_prompt_features()
        bank = self.model._select_diverse_features(labels=None)
        reference = []
        for c in range(text.shape[0]):
            z = self.model.diversity_augmentation(e, bank[c][None].expand(len(e), -1, -1))
            logits = self.model.logit_scale * (F.normalize(z, dim=-1) * text[c]).sum(-1)
            reference.append(logits)
        expected = torch.stack(reference, dim=-1)
        actual = self.model.classwise_tda_logits(e, text)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_class_chunking_is_exact(self):
        self.model.eval()
        e = self.model.causal_adapter(self.model.encode_image_features(self.images))
        text = self.model.encode_prompt_features()
        a = self.model.classwise_tda_logits(e, text)
        self.model.class_chunk_size = 3
        b = self.model.classwise_tda_logits(e, text)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_train_eval_scoring_agrees(self):
        self.model.eval()
        logits_train_formula = self.model(self.images, self.labels).interventional_logits
        logits_infer_formula = self.model.predict(self.images, use_tda=True)
        torch.testing.assert_close(
            logits_train_formula.mean(dim=1), logits_infer_formula, atol=2e-6, rtol=2e-6
        )

    def test_gradients_reach_tda_e_and_text(self):
        self.model.train()
        out = self.model(self.images, labels=self.labels)
        loss = cipt_loss(
            out.interventional_logits,
            out.causal_logits,
            out.spurious_logits,
            out.causal_features,
            out.spurious_features,
            self.labels,
            beta=4.0,
            gamma=5.0,
        ).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for param in [
            self.model.causal_adapter.weight,
            self.model.spurious_adapter.weight,
            self.model.diversity_augmentation.mapping.weight,
            self.model.text_param,
        ]:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all().item())
            self.assertGreater(param.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

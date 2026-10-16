"""CPU-only test of the CLIP+learnable-text, CE-only objective.

Mocks CLIP weights; this checks execution/gradients, not PACS accuracy.
Run from DCCL/DCCL: python -m unittest discover -s tests -p test_cipt_clip_ce.py
"""

import math
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

from test_cipt_env_kl import Environment


class TinyAlgorithm(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(3.0)), requires_grad=False)
        self.visual_weight = nn.Parameter(torch.eye(4), requires_grad=False)

    def encode_image(self, images):
        return images @ self.visual_weight


class TinyPromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, tokenize, prompt_length, prompt_init):
        super().__init__()
        self.context = nn.Parameter(torch.tensor([
            [0.8, 0.3, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ]))
        self.register_buffer("tokenized_prompts", torch.arange(len(class_names)))

    def forward(self):
        return self.context


class TinyTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, prompts, tokenized_prompts):
        return prompts


class ClipCEOnlyTests(unittest.TestCase):
    def build(self):
        hparams = dict(
            cipt_objective="clip_ce", cipt_enabled=True,
            cipt_clip_backbone="ViT-B/16", cipt_clip_path="unused",
            cipt_class_names=["dog", "horse"],
            cipt_prompt_length=16, cipt_prompt_init="a photo of a",
            optimizer="adam", lr=0.01, weight_decay=0.,
            # Deliberately nonzero: must be ignored in clip_ce mode.
            cipt_beta=4., cipt_gamma=5., cipt_env_kl_weight=.1,
            cipt_causal_contrastive_weight=1.5,
        )
        with mock.patch.dict(Environment.__init__.__globals__, {
            "Algorithm": TinyAlgorithm,
            "load_frozen_clip": lambda *_: (TinyCLIP(), lambda texts: None),
            "PromptLearner": TinyPromptLearner,
            "CLIPTextEncoder": TinyTextEncoder,
        }):
            return Environment((4,), 2, 2, hparams)

    def test_only_one_ce_only_prompt_context_updates(self):
        torch.manual_seed(7)
        model = self.build()
        for unused in ("causal_decomposition", "tda", "sigm", "text_features",
                       "prompt_selector", "supcon_loss", "proj_head"):
            self.assertFalse(hasattr(model, unused), unused)
        optimized = [p for g in model.optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(optimized), 1)
        self.assertIs(optimized[0], model.prompt_learner.context)

        images = torch.tensor([[1., 0., .2, 0.], [0., 1., 0., .2],
                               [1., .1, 0., 0.], [.1, 1., 0., 0.]])
        labels = torch.tensor([0, 1, 0, 1])
        predicted = model.predict(images)
        expected = F.cross_entropy(predicted, labels)
        original = model.prompt_learner.context.detach().clone()
        with mock.patch.object(F, "cross_entropy", wraps=F.cross_entropy) as ce:
            metrics = model.update([images[:2], images[2:]],
                                   [labels[:2], labels[2:]])
        self.assertEqual(ce.call_count, 1)
        self.assertEqual(tuple(ce.call_args.args[0].shape), (4, 2))
        self.assertAlmostEqual(metrics["total_loss"], expected.item(), places=6)
        self.assertAlmostEqual(metrics["cipt_cls_loss"], expected.item(), places=6)
        self.assertFalse(torch.equal(original, model.prompt_learner.context.detach()))
        self.assertGreater(model.prompt_learner.context.grad.abs().sum().item(), 0.)
        self.assertIsNone(model.clip_model.visual_weight.grad)
        self.assertIsNone(model.clip_model.logit_scale.grad)

    def test_eval_uses_same_raw_clip_features_and_class_text(self):
        model = self.build()
        images = torch.tensor([[2., 0., 0., 0.], [0., 3., 0., 0.]])
        visual = F.normalize(images, dim=-1)
        text = F.normalize(model.prompt_learner.context, dim=-1)
        expected_logits = 3.0 * (visual @ text.t())
        torch.testing.assert_close(model.predict_embed(images), visual)
        torch.testing.assert_close(model.predict(images), expected_logits)


if __name__ == "__main__":
    unittest.main()

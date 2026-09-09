import unittest

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_modules import (
    SafeDiversePromptSelector,
    TextDiversityAugmentation,
)


class PromptEffectTest(unittest.TestCase):
    def test_prompt_effect_path_matches_one_token_attention(self):
        torch.manual_seed(0)
        batch, prompts, dim = 3, 5, 8
        tda = TextDiversityAugmentation(dim, num_heads=1).eval()
        causal = torch.randn(batch, dim)
        text = torch.randn(prompts, dim)

        reference = tda(causal, text)
        effects = tda.prompt_effects(text)
        reconstructed = tda.apply_prompt_effects(causal, effects)

        self.assertEqual(tuple(effects.shape), (prompts, dim))
        self.assertTrue(
            torch.allclose(reference, reconstructed, atol=1e-6, rtol=1e-5)
        )


class SafeDiversePromptSelectorTest(unittest.TestCase):
    def test_shortlist_uses_clip_visual_text_similarity(self):
        selector = SafeDiversePromptSelector(k=1, candidate_count=2)
        text = torch.eye(4)
        visual = torch.stack((text[2], text[0]))

        indices, relevance = selector.shortlist(visual, text)

        self.assertEqual(tuple(indices.shape), (2, 2))
        self.assertEqual(indices[:, 0].tolist(), [2, 0])
        self.assertTrue(torch.allclose(relevance[:, 0], torch.ones(2)))

    def test_selector_returns_unique_per_sample_indices_and_gradients(self):
        torch.manual_seed(1)
        batch, prompts, classes, dim = 4, 7, 3, 8
        k, candidate_count = 2, 4
        tda = TextDiversityAugmentation(dim, num_heads=1)
        selector = SafeDiversePromptSelector(
            k=k,
            candidate_count=candidate_count,
        )

        visual = torch.randn(batch, dim)
        causal = torch.randn(batch, dim, requires_grad=True)
        text = torch.randn(prompts, dim)
        class_features = F.normalize(torch.randn(classes, dim), dim=-1)

        candidate_indices, relevance = selector.shortlist(
            visual, text
        )
        candidate_contexts = text[candidate_indices]
        candidate_z = tda(causal, candidate_contexts)

        with torch.no_grad():
            base_logits = torch.einsum(
                "bd,cd->bc", F.normalize(causal, dim=-1), class_features
            )
            candidate_logits = torch.einsum(
                "bld,cd->blc", F.normalize(candidate_z, dim=-1), class_features
            )
        local_indices, global_indices, metrics = selector.select(
            candidate_indices,
            relevance,
            causal,
            candidate_z,
            base_logits,
            candidate_logits,
        )
        selected_z = selector.batch_gather(candidate_z, local_indices)

        self.assertEqual(tuple(candidate_indices.shape), (batch, candidate_count))
        self.assertEqual(tuple(global_indices.shape), (batch, k))
        self.assertEqual(tuple(selected_z.shape), (batch, k, dim))
        for row in global_indices:
            self.assertEqual(torch.unique(row).numel(), k)
        self.assertGreaterEqual(global_indices.min().item(), 0)
        self.assertLess(global_indices.max().item(), prompts)
        self.assertIn("prompt_selector_js", metrics)
        self.assertIn("prompt_selector_safe_fraction", metrics)
        self.assertIn("prompt_selector_fallback_fraction", metrics)

        # A non-uniform linear probe avoids the zero-gradient symmetry of
        # summing all coordinates immediately after LayerNorm.
        probe = torch.arange(1, dim + 1, dtype=selected_z.dtype)
        (selected_z * probe).sum().backward()
        self.assertIsNotNone(causal.grad)
        self.assertIsNotNone(tda.attention.in_proj_weight.grad)
        self.assertGreater(
            tda.attention.in_proj_weight.grad[2 * dim :].abs().sum().item(),
            0.0,
        )

    def test_unsafe_prompts_are_used_only_as_minimum_js_fallbacks(self):
        selector = SafeDiversePromptSelector(k=3, candidate_count=4)
        candidate_indices = torch.tensor([[0, 1, 2, 3]])
        relevance = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        causal = torch.zeros(1, 3)
        candidate_z = torch.tensor(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [-1.0, 0.0, 0.0]]]
        )
        base_logits = torch.tensor([[3.0, 0.0]])
        candidate_logits = torch.tensor(
            [[[2.5, 0.0],   # safe
              [0.99, 1.0],  # unsafe, but closest to the base distribution
              [2.0, 0.0],   # safe
              [0.0, 3.0]]]  # unsafe and high JS
        )

        _, global_indices, metrics = selector.select(
            candidate_indices,
            relevance,
            causal,
            candidate_z,
            base_logits,
            candidate_logits,
        )

        self.assertEqual(set(global_indices[0].tolist()), {0, 1, 2})
        self.assertAlmostEqual(
            metrics["prompt_selector_safe_fraction"].item(), 2.0 / 3.0
        )
        self.assertAlmostEqual(
            metrics["prompt_selector_fallback_fraction"].item(), 1.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_modules import (
    CausalDecomposition,
    SafeDiversePromptSelector,
    TextDiversityAugmentation,
)


class CausalDecompositionTest(unittest.TestCase):
    def test_identity_initialization_preserves_normalized_clip_features(self):
        torch.manual_seed(0)
        batch, dim = 4, 8
        module = CausalDecomposition(dim).eval()
        visual = torch.randn(batch, dim)
        expected = F.normalize(visual.float(), dim=-1)

        causal, spurious = module(visual)

        identity = torch.eye(dim)
        zeros = torch.zeros(dim)
        self.assertTrue(torch.allclose(module.causal_adapter.weight, identity))
        self.assertTrue(torch.allclose(module.spurious_adapter.weight, identity))
        self.assertTrue(torch.allclose(module.causal_adapter.bias, zeros))
        self.assertTrue(torch.allclose(module.spurious_adapter.bias, zeros))
        self.assertTrue(torch.allclose(causal, expected, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(spurious, expected, atol=1e-6, rtol=1e-5))


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
    def test_selector_returns_unique_per_sample_indices_and_gradients(self):
        torch.manual_seed(1)
        batch, prompts, classes, dim = 4, 7, 3, 8
        k, candidate_count = 2, 4
        tda = TextDiversityAugmentation(dim, num_heads=1)
        selector = SafeDiversePromptSelector(
            k=k,
            candidate_count=candidate_count,
            causal_penalty=0.5,
            js_weight=1.0,
            diversity_weight=0.1,
        )

        causal = torch.randn(batch, dim, requires_grad=True)
        spurious = torch.randn(batch, dim, requires_grad=True)
        text = torch.randn(prompts, dim)
        class_features = F.normalize(torch.randn(classes, dim), dim=-1)

        effects = tda.prompt_effects(text)
        candidate_indices, relevance = selector.shortlist(
            causal, spurious, effects
        )
        candidate_effects = effects[candidate_indices]
        candidate_z = tda.apply_prompt_effects(causal, candidate_effects)

        with torch.no_grad():
            base_logits = torch.einsum(
                "bd,cd->bc", F.normalize(causal, dim=-1), class_features
            )
            candidate_logits = torch.einsum(
                "bld,cd->blc", F.normalize(candidate_z, dim=-1), class_features
            )
        local_indices, global_indices, metrics = selector.rerank(
            candidate_indices,
            relevance,
            candidate_effects,
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

        selected_z.sum().backward()
        self.assertIsNotNone(causal.grad)
        self.assertIsNotNone(tda.attention.in_proj_weight.grad)
        self.assertGreater(
            tda.attention.in_proj_weight.grad[2 * dim :].abs().sum().item(),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

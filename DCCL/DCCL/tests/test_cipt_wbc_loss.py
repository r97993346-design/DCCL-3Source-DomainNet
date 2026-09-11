import unittest

import torch

from domainbed.algorithms.cipt_losses import (
    WeakestDomainBridgeContrastiveLoss,
    weakest_domain_bridge_contrastive_loss,
)


class WeakestDomainBridgeContrastiveLossTest(unittest.TestCase):
    def test_well_connected_causal_features_have_lower_loss(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        domains = torch.tensor([0, 1, 2, 0, 1, 2])
        well_connected = torch.tensor(
            [
                [1.00, 0.00],
                [0.99, 0.10],
                [0.98, -0.10],
                [-1.00, 0.00],
                [-0.99, 0.10],
                [-0.98, -0.10],
            ],
            requires_grad=True,
        )
        broken_bridge = well_connected.detach().clone()
        broken_bridge[2] = torch.tensor([-0.80, 0.20])
        broken_bridge.requires_grad_(True)

        loss_module = WeakestDomainBridgeContrastiveLoss(
            topk=1,
            temperature=0.2,
            margin=0.1,
        )
        good_loss, good_metrics = loss_module(
            well_connected, labels, domains
        )
        bad_loss, bad_metrics = loss_module(
            broken_bridge, labels, domains
        )

        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertGreater(
            good_metrics["weakest_positive_similarity"].item(),
            bad_metrics["weakest_positive_similarity"].item(),
        )
        self.assertEqual(good_metrics["valid_anchor_fraction"].item(), 1.0)

        bad_loss.backward()
        self.assertIsNotNone(broken_bridge.grad)
        self.assertTrue(torch.isfinite(broken_bridge.grad).all())
        self.assertGreater(broken_bridge.grad.abs().sum().item(), 0.0)

    def test_same_domain_matches_are_not_cross_domain_bridges(self):
        causal = torch.randn(4, 5, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        domains = torch.tensor([0, 0, 1, 1])

        loss, metrics = weakest_domain_bridge_contrastive_loss(
            causal, labels, domains
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["valid_anchor_fraction"].item(), 0.0)
        self.assertEqual(metrics["domain_coverage_fraction"].item(), 0.0)

        loss.backward()
        self.assertIsNotNone(causal.grad)
        self.assertEqual(causal.grad.abs().sum().item(), 0.0)

    def test_partial_class_domain_groups_are_skipped_and_reported(self):
        causal = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, 0.1],
                [-0.9, -0.1],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 1])
        domains = torch.tensor([0, 1, 0, 1, 2])

        _, metrics = weakest_domain_bridge_contrastive_loss(
            causal, labels, domains, topk=2
        )

        self.assertEqual(metrics["valid_anchor_fraction"].item(), 1.0)
        self.assertAlmostEqual(
            metrics["domain_coverage_fraction"].item(), 0.8, places=6
        )

    def test_batch_without_different_class_has_no_valid_anchor(self):
        causal = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.8, -0.1]],
            requires_grad=True,
        )
        labels = torch.zeros(3, dtype=torch.long)
        domains = torch.tensor([0, 1, 2])

        loss, metrics = weakest_domain_bridge_contrastive_loss(
            causal, labels, domains
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["valid_anchor_fraction"].item(), 0.0)
        self.assertEqual(metrics["domain_coverage_fraction"].item(), 1.0)

        loss.backward()
        self.assertEqual(causal.grad.abs().sum().item(), 0.0)

    def test_invalid_hyperparameters_are_rejected(self):
        causal = torch.randn(2, 3)
        labels = torch.tensor([0, 1])
        domains = torch.tensor([0, 1])

        with self.assertRaises(ValueError):
            weakest_domain_bridge_contrastive_loss(
                causal, labels, domains, topk=0
            )
        with self.assertRaises(ValueError):
            weakest_domain_bridge_contrastive_loss(
                causal, labels, domains, temperature=0.0
            )
        with self.assertRaises(ValueError):
            weakest_domain_bridge_contrastive_loss(
                causal, labels, domains, margin=-0.1
            )
        with self.assertRaises(ValueError):
            WeakestDomainBridgeContrastiveLoss(eps=0.0)
        with self.assertRaises(ValueError):
            weakest_domain_bridge_contrastive_loss(
                torch.randn(2, 3, 4), labels, domains
            )
        with self.assertRaises(ValueError):
            weakest_domain_bridge_contrastive_loss(
                causal, labels[:1], domains
            )


if __name__ == "__main__":
    unittest.main()

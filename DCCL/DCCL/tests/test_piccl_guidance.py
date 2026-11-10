"""Regression checks for training-only PICCL guidance modes."""
import pytest

torch = pytest.importorskip("torch")

from domainbed import hparams_registry
from domainbed.algorithms.piccl import PICCLForwardModel
from domainbed.algorithms.piccl_components import InterventionSensitiveSubspace


@pytest.mark.parametrize("mode", ["guide_contrast", "replace_gt"])
def test_train_only_guidance_returns_original_features_at_inference(mode):
    featurizer = torch.nn.Linear(3, 8)
    classifier = torch.nn.Linear(8, 2)
    basis = InterventionSensitiveSubspace(8, 2)
    model = PICCLForwardModel(
        featurizer, basis, classifier, torch.tensor(0.20),
        guidance_mode=mode,
    )
    x = torch.randn(5, 3)
    assert torch.allclose(model.predict_embed(x), featurizer(x))
    assert torch.allclose(model.predict(x), classifier(featurizer(x)))


def test_v6_mediator_mode_still_uses_causal_projection():
    featurizer = torch.nn.Linear(3, 8)
    classifier = torch.nn.Linear(8, 2)
    basis = InterventionSensitiveSubspace(8, 2)
    model = PICCLForwardModel(
        featurizer, basis, classifier, torch.tensor(0.20),
        guidance_mode="mediator",
    )
    x = torch.randn(5, 3)
    z = featurizer(x)
    q = basis.orthonormal_basis(detach=True)
    expected = z - 0.20 * ((z @ q) @ q.T)
    assert torch.allclose(model.predict_embed(x), expected)


def test_registered_guidance_mode_keeps_existing_default():
    hparams = hparams_registry.default_hparams("PICCL", "PACS")
    assert hparams["piccl_guidance_mode"] == "mediator"

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DCCL"))

from domainbed.lib.rise_utils import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    clean_class_names,
    clip_kd_loss,
    get_prompt_templates,
    preprocess_for_clip,
    proto_alignment_loss,
)


def test_rise_prompt_template_counts():
    assert len(get_prompt_templates("simple")) == 1
    assert len(get_prompt_templates("multi")) == 5
    assert len(get_prompt_templates("domain_invariant")) == 7
    assert len(get_prompt_templates("rise80")) == 80


def test_rise_class_names_are_label_aligned_and_cleaned():
    assert clean_class_names(["baseball_bat", "airplane"], expected_num_classes=2) == [
        "baseball bat",
        "airplane",
    ]


def test_clip_preprocess_denormalizes_imagenet_before_clip_normalize():
    imagenet_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    clip_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    x01 = torch.full((2, 3, 4, 4), 0.5)
    x_student = (x01 - imagenet_mean) / imagenet_std

    x_clip = preprocess_for_clip(x_student)

    assert torch.allclose(x_clip, (x01 - clip_mean) / clip_std, atol=1e-6)


def test_rise_losses_are_finite_and_detach_teacher_inputs():
    student_logits = torch.randn(4, 3, requires_grad=True)
    teacher_logits = torch.randn(4, 3, requires_grad=True)
    kd = clip_kd_loss(student_logits, teacher_logits, temperature=2.0)
    assert torch.isfinite(kd)
    kd.backward(retain_graph=True)
    assert student_logits.grad is not None
    assert teacher_logits.grad is None

    features = torch.randn(4, 5, requires_grad=True)
    projector = torch.nn.Linear(5, 3)
    labels = torch.tensor([0, 1, 2, 1])
    text_prototypes = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1).requires_grad_()
    proto_loss, cosine_mean = proto_alignment_loss(features, labels, text_prototypes, projector)
    assert torch.isfinite(proto_loss)
    assert torch.isfinite(cosine_mean)
    proto_loss.backward()
    assert features.grad is not None
    assert text_prototypes.grad is None

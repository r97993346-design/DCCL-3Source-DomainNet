"""Thin DomainBed adapter around the vendored official CIPT package.

The CIPT model, losses, optimizer, and scheduler live in ``DCCL/DCCL/cipt`` and
are copied verbatim from the author-linked public repository.  This file only
adapts DomainBed's multi-domain ``update/predict`` interface and optional local
OpenAI-CLIP checkpoint loading; it does not reimplement CIPT internals.
"""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import Algorithm

# Use the repository-vendored OpenAI CLIP package so an explicit local
# checkpoint can be loaded without network access. This is integration glue;
# the vendored official CIPT source itself is left unchanged.
_BUNDLED_CLIP = Path(__file__).resolve().parents[4] / "CLIP"
if str(_BUNDLED_CLIP) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_CLIP))
import clip

from cipt import CIPTModel, cipt_loss, cosine_scheduler, make_optimizer


class CIPT(Algorithm):
    """DomainBed wrapper for the unchanged official CIPT implementation."""

    use_official_clip_preprocess = True
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        class_names = list(
            hparams.get(
                "cipt_class_names",
                ["class {}".format(i) for i in range(num_classes)],
            )
        )
        if len(class_names) != num_classes:
            raise ValueError(
                "CIPT class-name count {} != num_classes {}".format(
                    len(class_names), num_classes
                )
            )

        self.beta = float(hparams.get("cipt_beta", 4.0))
        self.gamma = float(hparams.get("cipt_gamma", 5.0))
        self.global_batch_size = int(hparams.get("cipt_global_batch_size", 64))
        self.epochs = int(hparams.get("cipt_epochs", 30))
        self.steps_per_epoch = max(
            1, int(hparams.get("cipt_steps_per_epoch", 1))
        )
        self.debug_shapes = bool(hparams.get("cipt_debug_shapes", False))
        self._num_updates = 0

        backbone = hparams.get("cipt_clip_backbone", "ViT-B/16")
        local_path = hparams.get("cipt_clip_path", "")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if local_path:
            local_path = Path(local_path).expanduser()
            if not local_path.is_file():
                raise FileNotFoundError(
                    "CIPT OpenAI CLIP checkpoint does not exist: {}".format(
                        local_path
                    )
                )
            clip_model, _ = clip.load(
                str(local_path), device=device, jit=False
            )
        else:
            clip_model, _ = clip.load(
                backbone, device=device, jit=False
            )

        self.model = CIPTModel(
            clip_model,
            classnames=class_names,
            tokenize=clip.tokenize,
            n_ctx=int(hparams.get("cipt_prompt_length", 16)),
            ctx_init=hparams.get("cipt_prompt_init", "a photo of a"),
            num_diverse_templates=int(hparams.get("cipt_k", 4)),
            num_heads=int(hparams.get("cipt_tda_heads", 8)),
        )

        lr = float(hparams.get("cipt_lr", 2.5e-3))
        weight_decay = float(hparams.get("cipt_weight_decay", 0.0))
        self.optimizer = make_optimizer(
            self.model, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = cosine_scheduler(
            self.optimizer, epochs=self.epochs, min_lr=0.0
        )

        print(
            "Official CIPT source: ckghostwj/CIPT@a805d878acc7d79778d1ec1c1e4d73ba6aff334b; "
            "backbone={}, beta={}, gamma={}, K={}, heads={}, lr={}, wd={}, "
            "shots={}, global_batch={}, epochs={}, steps_per_epoch={}".format(
                backbone,
                self.beta,
                self.gamma,
                int(hparams.get("cipt_k", 4)),
                int(hparams.get("cipt_tda_heads", 8)),
                lr,
                weight_decay,
                int(hparams.get("cipt_shots", 16)),
                self.global_batch_size,
                self.epochs,
                self.steps_per_epoch,
            )
        )

    def _trim_merged_domainbed_batch(self, images, labels):
        """Adapt DomainBed's merged source minibatches to the requested batch."""
        if images.shape[0] <= self.global_batch_size:
            return images, labels
        indices = torch.randperm(images.shape[0], device=images.device)[
            : self.global_batch_size
        ]
        return images.index_select(0, indices), labels.index_select(0, indices)

    def update(self, x, y, **kwargs):
        images = torch.cat(x)
        labels = torch.cat(y)
        images, labels = self._trim_merged_domainbed_batch(images, labels)

        output = self.model(images, labels)
        if output.interventional_logits is None:
            raise RuntimeError(
                "Official CIPT returned no interventional logits during training."
            )

        losses = cipt_loss(
            output.interventional_logits,
            output.causal_logits,
            output.spurious_logits,
            output.causal_features,
            output.spurious_features,
            labels,
            beta=self.beta,
            gamma=self.gamma,
        )

        self.optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        self.optimizer.step()

        self._num_updates += 1
        if self._num_updates % self.steps_per_epoch == 0:
            self.scheduler.step()

        if self.debug_shapes:
            print(
                "CIPT official shapes: image={} e={} s={} z={} logits={} effective_batch={}".format(
                    tuple(output.image_features.shape),
                    tuple(output.causal_features.shape),
                    tuple(output.spurious_features.shape),
                    tuple(output.augmented_features.shape),
                    tuple(output.interventional_logits.shape),
                    int(images.shape[0]),
                )
            )

        return {
            "loss": losses.loss.item(),
            "total_loss": losses.loss.item(),
            "cipt_cls_loss": losses.classification.item(),
            "cipt_de_loss": losses.decomposition.item(),
            "cipt_ind_loss": losses.independence.item(),
            "cipt_causal_ce": losses.causal_ce.item(),
            "cipt_spurious_kl": losses.spurious_kl.item(),
            "cipt_lr": self.optimizer.param_groups[0]["lr"],
            "cipt_effective_batch": float(images.shape[0]),
            "mean_e_norm": output.causal_features.norm(dim=-1).mean().item(),
            "mean_s_norm": output.spurious_features.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                output.causal_features, output.spurious_features, dim=-1
            ).mean().item(),
            "ce_loss": losses.classification.item(),
            "sup_cl_loss": 0.0,
            "pre_cl_loss": 0.0,
        }

    @torch.no_grad()
    def predict(self, x):
        return self.model.predict(x, use_tda=True)

    @torch.no_grad()
    def predict_embed(self, x):
        image_features = self.model.encode_image_features(x)
        return self.model.causal_adapter(image_features)

"""Official-aligned CIPT integrated with DCCL causal-space contrastive learning.

CIPT components follow the public CIPT implementation:
- frozen OpenAI CLIP
- normalized image embeddings
- identity-initialized causal/spurious single-linear adapters
- learnable CoOp-style class prompts
- class-conditioned ImageNet template bank with random K-template sampling
- TDA cross-attention
- official decomposition, independence and intervention classification losses

DCCL integration is intentionally limited to the existing causal-space SupCon,
pretrained anchoring and vector regularizer used by feature/multiprompt.
"""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import Algorithm, SupConLoss
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)
from domainbed.algorithms.cipt_modules import (
    CausalDecomposition,
    TextDiversityAugmentation,
)
from domainbed.algorithms.cipt_prompt import CIPTTextFeatures, load_frozen_clip


class CIPTDCCL(Algorithm):
    """Official-aligned CIPT + DCCL contrastive learning on causal features."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if not hparams["cipt_enabled"]:
            raise ValueError("CIPTDCCL requires cipt_enabled=true")

        self.clip_model, tokenize = load_frozen_clip(
            hparams["cipt_clip_backbone"], hparams["cipt_clip_path"]
        )
        dim = int(self.clip_model.visual.output_dim)
        class_names = hparams.get("cipt_class_names") or [
            "class {}".format(i) for i in range(num_classes)
        ]

        # Official CIPT core modules.
        self.causal_decomposition = CausalDecomposition(dim, identity_init=True)
        self.text_features = CIPTTextFeatures(
            class_names,
            self.clip_model,
            tokenize,
            hparams["cipt_prompt_length"],
            hparams["cipt_prompt_init"],
            hparams["cipt_k"],
            sample_templates=True,
        )
        self.tda = TextDiversityAugmentation(
            dim, hparams["cipt_tda_heads"], dropout=0.0
        )

        # DCCL integration modules retained from feature/multiprompt.
        hidden = dim // 4
        if hparams["n_layer"] == 1:
            self.proj_head = nn.Sequential(nn.Linear(dim, hidden))
            self.pre_proj_head = nn.Sequential(nn.Linear(dim, hidden))
        else:
            self.proj_head = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.pre_proj_head = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        # Vector analogue of the feature/multiprompt DCCL representation regularizer.
        self.reg_log_variance = nn.Parameter(torch.full((1, dim), -2.2521685))
        self.supcon_loss = SupConLoss(hparams["t"])
        self.supcon_loss_pre = SupConLoss(hparams["t_pre"])

        self.beta = hparams["cipt_beta"]
        self.gamma = hparams["cipt_gamma"]
        self.contrastive_weight = hparams["cipt_contrastive_weight"]
        self.l_d = hparams["l_d"]
        self.l_layer = hparams["l_layer"]
        self.debug_shapes = bool(hparams.get("cipt_debug_shapes", False))

        # Official CIPT uses Adam with lr=2.5e-3 and zero weight decay for
        # prompt/adapters/TDA. Keep DCCL integration parameters on the existing
        # DomainBed/DCCL learning rate so the added contrastive branch is not
        # accidentally trained at the much larger prompt-tuning rate.
        cipt_lr = float(hparams.get("cipt_lr", 2.5e-3))
        cipt_weight_decay = float(hparams.get("cipt_weight_decay", 0.0))

        cipt_params = []
        cipt_params.extend(self.causal_decomposition.parameters())
        cipt_params.extend(self.text_features.prompt_learner.parameters())
        cipt_params.extend(self.tda.parameters())
        cipt_params = [p for p in cipt_params if p.requires_grad]

        dccl_params = []
        dccl_params.extend(self.proj_head.parameters())
        dccl_params.extend(self.pre_proj_head.parameters())
        dccl_params.append(self.reg_log_variance)
        dccl_params = [p for p in dccl_params if p.requires_grad]

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": cipt_params,
                    "lr": cipt_lr,
                    "weight_decay": cipt_weight_decay,
                },
                {
                    "params": dccl_params,
                    "lr": hparams["lr"],
                    "weight_decay": hparams["weight_decay"],
                },
            ]
        )

        # Official CIPT uses cosine LR decay. DomainBed/DCCL is step-based rather
        # than epoch-based, so use the mathematically equivalent cosine curve on
        # the configured training-step horizon for the CIPT parameter group only.
        # The DCCL integration group deliberately keeps its original constant LR.
        self.cipt_schedule_steps = max(
            1, int(hparams.get("cipt_schedule_steps", 1))
        )

        def cipt_cosine_factor(step):
            progress = min(float(step), float(self.cipt_schedule_steps)) / float(
                self.cipt_schedule_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[cipt_cosine_factor, lambda _step: 1.0],
        )

        self.trainable_parameter_count = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        self.frozen_parameter_count = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        print(
            "CIPTDCCL parameters: trainable={}, frozen={}".format(
                self.trainable_parameter_count, self.frozen_parameter_count
            )
        )
        print(
            "CIPTDCCL trainable tensors: {}".format(
                ", ".join(
                    name
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                )
            )
        )
        print(
            "CIPTDCCL optimizer: cipt_lr={}, dccl_lr={}, cosine_steps={}".format(
                cipt_lr, hparams["lr"], self.cipt_schedule_steps
            )
        )

    def train(self, mode=True):
        super().train(mode)
        # Frozen CLIP must remain in inference mode while prompt/adapters train.
        self.clip_model.eval()
        self.text_features.text_encoder.eval()
        return self

    def _visual(self, images):
        """Official CIPT image path: frozen CLIP followed by L2 normalization."""
        with torch.no_grad():
            visual = self.clip_model.encode_image(images).float()
        return F.normalize(visual, dim=-1)

    def _logits(self, embeddings, class_features):
        scale = self.clip_model.logit_scale.exp().detach().float()
        embeddings = F.normalize(embeddings.float(), dim=-1)
        class_features = F.normalize(class_features.float(), dim=-1)
        if embeddings.ndim == 2:
            return scale * embeddings @ class_features.t()
        if embeddings.ndim == 3:
            return scale * torch.einsum("bkd,cd->bkc", embeddings, class_features)
        raise ValueError(
            "Expected embeddings [B,D] or [B,K,D], got {}".format(
                tuple(embeddings.shape)
            )
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x_aug = torch.cat(kwargs["x_2"])
        labels = torch.cat(y)

        visual = self._visual(all_x)
        visual_aug = self._visual(all_x_aug)
        causal, spurious = self.causal_decomposition(visual)
        causal_aug, _ = self.causal_decomposition(visual_aug)

        # ----- Official CIPT objective on the original view -----
        class_features = self.text_features.class_features()
        causal_logits = self._logits(causal, class_features)
        spurious_logits = self._logits(spurious, class_features)
        loss_de = cipt_decomposition_loss(causal_logits, spurious_logits, labels)
        loss_ind = cipt_independence_loss(causal, spurious)

        diverse_features = self.text_features.select_diverse_features(labels=labels)
        interventions = self.tda(causal, diverse_features)
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        # ----- DCCL extension: contrast causal representations only -----
        projected = self.proj_head(causal)
        projected_aug = self.proj_head(causal_aug)
        contrast_features = torch.stack(
            (
                F.normalize(projected, dim=-1),
                F.normalize(projected_aug, dim=-1),
            ),
            dim=1,
        )
        loss_contrastive = self.supcon_loss(contrast_features, labels)

        # Retain feature/multiprompt's DCCL-inspired anchoring/regularization.
        pre_features = torch.stack(
            (
                F.normalize(self.pre_proj_head(causal), dim=-1),
                F.normalize(self.pre_proj_head(visual.detach()), dim=-1),
            ),
            dim=1,
        )
        pre_cl_loss = (
            self.supcon_loss_pre(pre_features, labels)
            if self.l_layer
            else causal.new_zeros(())
        )

        variance = F.softplus(self.reg_log_variance) + 1e-5
        reg_loss = (
            (
                ((causal - visual.detach()).pow(2) / variance)
                + variance.log()
            ).mean()
            / 2
            if self.l_d
            else causal.new_zeros(())
        )

        total = (
            loss_cls
            + self.beta * loss_de
            + self.gamma * loss_ind
            + self.contrastive_weight * loss_contrastive
            + self.l_layer * pre_cl_loss
            + self.l_d * reg_loss
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL shapes: v={} e={} s={} projected_e={} z_k={} "
                "diverse={} text_features={} logits={}".format(
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
                    tuple(projected.shape),
                    tuple(interventions.shape),
                    tuple(diverse_features.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_ind_loss": loss_ind.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "pre_cl_loss": pre_cl_loss.item(),
            "reg_loss": reg_loss.item(),
            "cipt_lr": self.optimizer.param_groups[0]["lr"],
            "dccl_lr": self.optimizer.param_groups[1]["lr"],
            "mean_v_norm": visual.norm(dim=-1).mean().item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    @torch.no_grad()
    def predict(self, x):
        """Official CIPT class-conditioned TDA inference, averaged over K."""
        was_training = self.training
        self.eval()

        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        diverse_features = self.text_features.select_diverse_features(labels=None)

        # diverse_features: [C,K,D]. Score every candidate class with its own
        # intervention contexts, matching the official CIPT inference path.
        num_classes, num_templates, dim = diverse_features.shape
        batch = causal.shape[0]
        causal_flat = causal[:, None, :].expand(batch, num_classes, dim).reshape(
            batch * num_classes, dim
        )
        diverse_flat = (
            diverse_features[None, :, :, :]
            .expand(batch, num_classes, num_templates, dim)
            .reshape(batch * num_classes, num_templates, dim)
        )
        z = self.tda(causal_flat, diverse_flat.float()).reshape(
            batch, num_classes, num_templates, dim
        )
        z = F.normalize(z.float(), dim=-1)
        text = F.normalize(class_features.float(), dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        logits = scale * torch.einsum("bckd,cd->bck", z, text).mean(dim=-1)

        self.train(was_training)
        return logits

    def predict_embed(self, x):
        causal, _ = self.causal_decomposition(self._visual(x))
        return causal

    def get_forward_model(self):
        # The package-level CIPTDCCL routes through cipt_dccl_official.py, which
        # overrides this with a SWAD-safe lightweight inference wrapper.
        return copy.deepcopy(self)

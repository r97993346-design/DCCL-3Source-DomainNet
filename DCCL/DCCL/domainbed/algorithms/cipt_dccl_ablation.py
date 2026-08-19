"""CIPT dual-prompt causal mutual-learning ablation.

This branch combines:
- shallow learnable visual prompt tokens P_v on a frozen CLIP ViT,
- the existing learnable class-text prompt tokens P_t,
- shared learnable B5c diversity tokens P_d with fixed B5c semantic suffixes,
- MetaPrompt-style asymmetric frozen cross-modal anchors,
- visual/text predictive mutual consistency,
- B5c geometry-preserving diversity regularization.

No episodic meta-learning is introduced. With cipt_pure=True, DCCL-side losses
remain disabled so the prompt/causal mechanism can be evaluated cleanly.
"""

import torch
from torch import nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)
from domainbed.algorithms.cipt_prompt import (
    B5C_GENERIC_EXPANDED_TEMPLATES,
    clip,
)
from domainbed.algorithms.cipt_visual_prompt import (
    VisualPromptLearner,
    encode_image_with_visual_prompt,
)
from domainbed.optimizers import get_optimizer


class DiversityPromptLearner(nn.Module):
    """Shared learnable tokens prepended to fixed B5c style/context semantics."""

    def __init__(self, clip_model, prompt_length=4):
        super().__init__()
        if int(prompt_length) < 1:
            raise ValueError("cipt_diversity_prompt_length must be >= 1")

        dtype = clip_model.dtype
        width = int(clip_model.ln_final.weight.shape[0])
        self.prompt_length = int(prompt_length)

        context = torch.empty(self.prompt_length, width, dtype=dtype)
        nn.init.normal_(context, std=0.02)
        self.context = nn.Parameter(context)

        tokenized = clip.tokenize(
            [
                "X " * self.prompt_length + template
                for template in B5C_GENERIC_EXPANDED_TEMPLATES
            ]
        )
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1])
        self.register_buffer(
            "token_suffix",
            embedding[:, 1 + self.prompt_length :],
        )
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self, indices):
        prefix = self.token_prefix.index_select(0, indices)
        suffix = self.token_suffix.index_select(0, indices)
        tokenized = self.tokenized_prompts.index_select(0, indices)
        context = self.context.unsqueeze(0).expand(prefix.shape[0], -1, -1)
        prompts = torch.cat((prefix, context, suffix), dim=1)
        return prompts, tokenized


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT with mutually constrained visual, class-text and diversity prompts."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.cipt_pure = bool(hparams.get("cipt_pure", True))
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5c")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        self.causal_consistency_weight = float(
            hparams.get("cipt_causal_consistency_weight", 1.0)
        )

        # P_v: learnable visual tokens, CLIP visual weights remain frozen.
        self.visual_prompt_enabled = bool(
            hparams.get("cipt_visual_prompt_enabled", True)
        )
        self.visual_prompt_length = int(
            hparams.get("cipt_visual_prompt_length", 4)
        )
        self.visual_prompt = None
        if self.visual_prompt_enabled:
            self.visual_prompt = VisualPromptLearner(
                self.clip_model,
                prompt_length=self.visual_prompt_length,
            )

        # P_t: the existing CoOp-style class prompt can be independently ablated.
        self.text_prompt_trainable = bool(
            hparams.get("cipt_text_prompt_trainable", True)
        )
        self.text_features.prompt_learner.context.requires_grad_(
            self.text_prompt_trainable
        )

        # P_d: one shared learnable context for all fixed B5c semantic suffixes.
        self.diversity_prompt_enabled = bool(
            hparams.get("cipt_diversity_prompt_enabled", True)
        )
        self.diversity_prompt_length = int(
            hparams.get("cipt_diversity_prompt_length", 4)
        )
        self.diversity_prompt = None
        if self.diversity_prompt_enabled:
            self.diversity_prompt = DiversityPromptLearner(
                self.clip_model,
                prompt_length=self.diversity_prompt_length,
            )
        self._last_b5c_indices = None

        # Frozen class-text teacher: no learnable context tokens.
        class_texts = [
            "a photo of a {}.".format(class_name)
            for class_name in self.text_features.class_names
        ]
        with torch.no_grad():
            tokens = clip.tokenize(class_texts).to(
                next(self.clip_model.parameters()).device
            )
            frozen_class_text = self.clip_model.encode_text(tokens).float()
            frozen_class_text = F.normalize(frozen_class_text, dim=-1)
        self.register_buffer(
            "frozen_class_text_bank",
            frozen_class_text,
        )

        self.ac_weight = float(hparams.get("cipt_ac_weight", 0.1))
        self.mutual_weight = float(hparams.get("cipt_mutual_weight", 0.05))
        self.diversity_weight = float(
            hparams.get("cipt_diversity_weight", 0.05)
        )
        self.mutual_temperature = float(
            hparams.get("cipt_mutual_temperature", 1.0)
        )
        if self.mutual_temperature <= 0:
            raise ValueError("cipt_mutual_temperature must be > 0")

        # Pure mode isolates CIPT + dual prompt from DCCL objectives.
        if self.cipt_pure:
            self.contrastive_weight = 0.0
            self.l_layer = 0.0
            self.l_d = 0.0

        # Base optimizer predates P_v/P_d and the P_t freeze switch.
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            trainable,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.trainable_parameter_count = sum(p.numel() for p in trainable)
        self.frozen_parameter_count = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )

        print(
            "CIPT dual-prompt: pure={}, mode={}, visual_prompt={}({}), "
            "text_prompt_trainable={}, diversity_prompt={}({}), "
            "ac_w={}, mutual_w={}, div_w={}, mutual_T={}".format(
                self.cipt_pure,
                self.cipt_template_mode,
                self.visual_prompt_enabled,
                self.visual_prompt_length if self.visual_prompt_enabled else 0,
                self.text_prompt_trainable,
                self.diversity_prompt_enabled,
                self.diversity_prompt_length
                if self.diversity_prompt_enabled
                else 0,
                self.ac_weight,
                self.mutual_weight,
                self.diversity_weight,
                self.mutual_temperature,
            )
        )
        print(
            "CIPT dual-prompt parameters: trainable={}, frozen={}".format(
                self.trainable_parameter_count,
                self.frozen_parameter_count,
            )
        )

    def _visual_anchor(self, images):
        """Unprompted frozen CLIP visual feature used as a stable teacher."""
        with torch.no_grad():
            return self.clip_model.encode_image(images).float()

    def _visual(self, images):
        """Prompted visual feature used by causal decomposition."""
        if not self.visual_prompt_enabled:
            return self._visual_anchor(images)
        return encode_image_with_visual_prompt(
            self.clip_model,
            images,
            self.visual_prompt,
        )

    def _learnable_b5c_features(self, indices):
        prompts, tokenized = self.diversity_prompt(indices)
        features = self.text_features.text_encoder(prompts, tokenized)
        return F.normalize(features.float(), dim=-1)

    def _intervention_features(self, labels=None):
        if (
            self.cipt_template_mode == "b5c"
            and self.diversity_prompt_enabled
        ):
            bank = self.text_features.b5c_text_bank
            indices = self.text_features._select_indices(
                bank.shape[0],
                bank.device,
            )
            self._last_b5c_indices = indices.detach()
            return self._learnable_b5c_features(indices)

        self._last_b5c_indices = None
        if self.cipt_template_mode == "b5b":
            return self.text_features.intervention_features(labels=labels)
        return self.text_features.irrelevant_text_features

    def _diversity_geometry_loss(self, diversity_features):
        """Keep P_d-adapted B5c pairwise geometry close to frozen CLIP B5c."""
        if (
            self.cipt_template_mode != "b5c"
            or not self.diversity_prompt_enabled
            or self._last_b5c_indices is None
        ):
            return diversity_features.new_zeros(())

        frozen = self.text_features.b5c_text_bank.index_select(
            0,
            self._last_b5c_indices,
        )
        frozen = F.normalize(frozen.detach(), dim=-1)
        learned = F.normalize(diversity_features, dim=-1)

        frozen_gram = frozen @ frozen.t()
        learned_gram = learned @ learned.t()
        return F.mse_loss(learned_gram, frozen_gram)

    @staticmethod
    def _js_divergence(logits_a, logits_b, temperature=1.0):
        """Symmetric JS divergence between class-prediction distributions."""
        p = F.softmax(logits_a / temperature, dim=-1)
        q = F.softmax(logits_b / temperature, dim=-1)
        m = 0.5 * (p + q)
        log_m = m.clamp_min(1e-8).log()
        return 0.5 * (
            F.kl_div(log_m, p, reduction="batchmean")
            + F.kl_div(log_m, q, reduction="batchmean")
        )

    def _prompt_regularizers(
        self,
        causal,
        visual_anchor,
        class_features,
        labels,
        diversity_features,
    ):
        """MetaPrompt-style asymmetric anchors plus mutual/diversity losses."""
        frozen_text = F.normalize(
            self.frozen_class_text_bank.float(),
            dim=-1,
        )

        # P_v -> causal e must remain compatible with frozen text semantics.
        visual_to_text_logits = self._logits(
            causal[:, None, :],
            frozen_text,
        )[:, 0]
        loss_ac_visual = F.cross_entropy(
            visual_to_text_logits,
            labels,
        )

        # P_t is trained against the unprompted frozen visual representation.
        text_to_visual_logits = self._logits(
            visual_anchor[:, None, :],
            class_features,
        )[:, 0]
        loss_ac_text = F.cross_entropy(
            text_to_visual_logits,
            labels,
        )
        loss_ac = 0.5 * (loss_ac_visual + loss_ac_text)

        # The two independently anchored prompt predictions should agree.
        loss_mutual = self._js_divergence(
            visual_to_text_logits,
            text_to_visual_logits,
            temperature=self.mutual_temperature,
        )

        # P_d may adapt but must not destroy B5c's relative semantic diversity.
        loss_diversity = self._diversity_geometry_loss(
            diversity_features,
        )

        return (
            loss_ac,
            loss_ac_visual,
            loss_ac_text,
            loss_mutual,
            loss_diversity,
        )

    def _update_pure(self, x, y):
        """Pure CIPT + P_v/P_t/P_d path with no DCCL-side losses."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        visual = self._visual(all_x)
        visual_anchor = (
            self._visual_anchor(all_x)
            if self.visual_prompt_enabled
            else visual.detach()
        )
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()

        causal_logits = self._logits(
            causal[:, None, :],
            class_features,
        )[:, 0]
        spurious_logits = self._logits(
            spurious[:, None, :],
            class_features,
        )[:, 0]
        loss_de = cipt_decomposition_loss(
            causal_logits,
            spurious_logits,
            labels,
        )
        loss_ind = cipt_independence_loss(causal, spurious)

        diversity_features = self._intervention_features(labels=labels)
        interventions = self.tda(causal, diversity_features)
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        (
            loss_ac,
            loss_ac_visual,
            loss_ac_text,
            loss_mutual,
            loss_diversity,
        ) = self._prompt_regularizers(
            causal,
            visual_anchor,
            class_features,
            labels,
            diversity_features,
        )

        total = (
            loss_cls
            + self.beta * loss_de
            + self.gamma * loss_ind
            + self.ac_weight * loss_ac
            + self.mutual_weight * loss_mutual
            + self.diversity_weight * loss_diversity
        )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        zero = causal.new_zeros(())
        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de.item(),
            "cipt_de_aug_loss": zero.item(),
            "cipt_ind_loss": loss_ind.item(),
            "prompt_ac_loss": loss_ac.item(),
            "prompt_ac_visual_loss": loss_ac_visual.item(),
            "prompt_ac_text_loss": loss_ac_text.item(),
            "prompt_mutual_loss": loss_mutual.item(),
            "prompt_diversity_loss": loss_diversity.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": zero.item(),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal,
                spurious,
                dim=-1,
            ).mean().item(),
        }

    def _update_fusion(self, x, y, x_2):
        """Dual prompt CIPT plus the existing optional DCCL two-view losses."""
        all_x = torch.cat(x)
        all_x_aug = torch.cat(x_2)
        labels = torch.cat(y)

        visual = self._visual(all_x)
        visual_aug = self._visual(all_x_aug)
        visual_anchor = (
            self._visual_anchor(all_x)
            if self.visual_prompt_enabled
            else visual.detach()
        )

        causal, spurious = self.causal_decomposition(visual)
        causal_aug, spurious_aug = self.causal_decomposition(visual_aug)
        class_features = self.text_features.class_features()

        causal_logits = self._logits(
            causal[:, None, :],
            class_features,
        )[:, 0]
        spurious_logits = self._logits(
            spurious[:, None, :],
            class_features,
        )[:, 0]
        loss_de_orig = cipt_decomposition_loss(
            causal_logits,
            spurious_logits,
            labels,
        )

        causal_logits_aug = self._logits(
            causal_aug[:, None, :],
            class_features,
        )[:, 0]
        spurious_logits_aug = self._logits(
            spurious_aug[:, None, :],
            class_features,
        )[:, 0]
        loss_de_aug = cipt_decomposition_loss(
            causal_logits_aug,
            spurious_logits_aug,
            labels,
        )
        loss_de = 0.5 * (loss_de_orig + loss_de_aug)
        loss_ind = cipt_independence_loss(causal, spurious)

        loss_causal_consistency = (
            1.0 - F.cosine_similarity(causal, causal_aug, dim=-1)
        ).mean()

        diversity_features = self._intervention_features(labels=labels)
        interventions = self.tda(causal, diversity_features)
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        (
            loss_ac,
            loss_ac_visual,
            loss_ac_text,
            loss_mutual,
            loss_diversity,
        ) = self._prompt_regularizers(
            causal,
            visual_anchor,
            class_features,
            labels,
            diversity_features,
        )

        projected = self.proj_head(causal)
        projected_aug = self.proj_head(causal_aug)
        contrast_features = torch.stack(
            (
                F.normalize(projected, dim=-1),
                F.normalize(projected_aug, dim=-1),
            ),
            dim=1,
        )
        loss_contrastive = self.supcon_loss(
            contrast_features,
            labels,
        )

        pre_features = torch.stack(
            (
                F.normalize(self.pre_proj_head(causal), dim=-1),
                F.normalize(self.pre_proj_head(visual_anchor), dim=-1),
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
                (causal - visual_anchor).pow(2) / variance
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
            + self.ac_weight * loss_ac
            + self.mutual_weight * loss_mutual
            + self.diversity_weight * loss_diversity
            + self.causal_consistency_weight * loss_causal_consistency
            + self.contrastive_weight * loss_contrastive
            + self.l_layer * pre_cl_loss
            + self.l_d * reg_loss
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL dual-prompt shapes: mode={} v={} v_aug={} "
                "v_anchor={} e={} e_aug={} s={} s_aug={} z_k={} "
                "text_features={} logits={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(visual_aug.shape),
                    tuple(visual_anchor.shape),
                    tuple(causal.shape),
                    tuple(causal_aug.shape),
                    tuple(spurious.shape),
                    tuple(spurious_aug.shape),
                    tuple(interventions.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de_orig.item(),
            "cipt_de_aug_loss": loss_de_aug.item(),
            "cipt_ind_loss": loss_ind.item(),
            "prompt_ac_loss": loss_ac.item(),
            "prompt_ac_visual_loss": loss_ac_visual.item(),
            "prompt_ac_text_loss": loss_ac_text.item(),
            "prompt_mutual_loss": loss_mutual.item(),
            "prompt_diversity_loss": loss_diversity.item(),
            "causal_consistency_loss": loss_causal_consistency.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "pre_cl_loss": pre_cl_loss.item(),
            "reg_loss": reg_loss.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal,
                spurious,
                dim=-1,
            ).mean().item(),
        }

    def update(self, x, y, **kwargs):
        if self.cipt_pure:
            return self._update_pure(x, y)

        if "x_2" not in kwargs:
            raise KeyError(
                "CIPTDCCL fusion mode requires x_2: one original image and "
                "one augmented view per sample."
            )
        return self._update_fusion(x, y, kwargs["x_2"])

    def predict(self, x):
        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()

        if self.cipt_template_mode != "b5b":
            diverse_features = self._intervention_features(labels=None)
            interventions = self.tda(causal, diverse_features)
            return self._logits(interventions, class_features).mean(dim=1)

        diverse_features = self.text_features.intervention_features(labels=None)
        num_classes, num_templates, dim = diverse_features.shape
        batch = causal.shape[0]
        causal_flat = (
            causal[:, None, :]
            .expand(batch, num_classes, dim)
            .reshape(batch * num_classes, dim)
        )
        context_flat = (
            diverse_features[None, :, :, :]
            .expand(batch, num_classes, num_templates, dim)
            .reshape(batch * num_classes, num_templates, dim)
        )
        z = self.tda(causal_flat, context_flat).reshape(
            batch,
            num_classes,
            num_templates,
            dim,
        )
        z = F.normalize(z, dim=-1)
        text = F.normalize(class_features, dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        return scale * torch.einsum(
            "bckd,cd->bck",
            z,
            text,
        ).mean(dim=-1)

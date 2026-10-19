"""Class-agnostic text perturbations train E with optional fused CE and MCC.

``cipt_objective=legacy`` delegates to the original component-ablation wrapper.
``env_kl`` uses explicit E CE, S uniformity, E/S independence, original-E
contrastive learning, optional fused-feature label CE, and symmetric prompt KL.
Fused CE and prompt KL have independent weights; inference is configurable.
"""

import math

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import Algorithm
from domainbed.algorithms.cipt_dccl_component_ablation import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_prompt import CLIPTextEncoder, PromptLearner, load_frozen_clip
from domainbed.algorithms.cipt_sigm import SemanticInteractionGuidance
from domainbed.optimizers import get_optimizer
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss_terms,
    independence_loss,
    prompt_consistency_loss,
)


class CIPTDCCL(_BaseCIPTDCCL):
    """Legacy or env-KL classifier with switchable text fusion and inference."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        objective = str(hparams.get("cipt_objective", "legacy")).lower()
        if objective not in ("legacy", "env_kl", "clip_ce"):
            raise ValueError("cipt_objective must be legacy, env_kl, or clip_ce")

        if objective == "clip_ce":
            # Minimal CoOp-style baseline. Crucially, do NOT call the CIPTDCCL
            # parent constructor: it would create E/S, TDA, SIGM, selectors,
            # projection heads and contrastive objectives unnecessarily.
            Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
            self.objective_mode = objective
            if not hparams.get("cipt_enabled", True):
                raise ValueError("clip_ce requires cipt_enabled=true")
            self.clip_model, tokenize = load_frozen_clip(
                hparams["cipt_clip_backbone"], hparams["cipt_clip_path"]
            )
            class_names = hparams.get("cipt_class_names") or [
                "class {}".format(i) for i in range(num_classes)
            ]
            if len(class_names) != num_classes:
                raise ValueError("cipt_class_names must match num_classes")
            self.prompt_learner = PromptLearner(
                class_names, self.clip_model, tokenize,
                hparams["cipt_prompt_length"], hparams["cipt_prompt_init"],
            )
            self.text_encoder = CLIPTextEncoder(self.clip_model)
            # Only the learnable text context enters the optimizer.
            self.optimizer = get_optimizer(
                hparams["optimizer"], self.prompt_learner.parameters(),
                lr=hparams["lr"], weight_decay=hparams["weight_decay"],
            )
            self.trainable_parameter_count = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.frozen_parameter_count = sum(
                p.numel() for p in self.parameters() if not p.requires_grad
            )
            print("CIPTDCCL clip_ce: raw CLIP visual + learnable class prompts; "
                  "only CE, weight=1, trainable={}, frozen={}".format(
                      self.trainable_parameter_count, self.frozen_parameter_count))
            return

        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.objective_mode = objective
        if self.objective_mode == "legacy":
            return

        self.fusion_mode = str(hparams.get("cipt_fusion_mode", "tda")).lower()
        self.inference_mode = str(hparams.get("cipt_env_inference", "e_only")).lower()
        if self.fusion_mode not in ("tda", "sigm"):
            raise ValueError("cipt_fusion_mode must be tda or sigm")
        if self.inference_mode not in ("e_only", "mean_logits", "mean_probs"):
            raise ValueError("cipt_env_inference must be e_only, mean_logits, or mean_probs")

        if self.fusion_mode == "sigm":
            # Preserve the existing _select_interventions selector exactly:
            # it calls self.tda(E, text) for each candidate/prompt. Swap only
            # the fusion operator; both training and optional fused inference
            # use the same module without changing the selector or loss.
            dim = self.causal_decomposition.causal_adapter.out_features
            self.tda = SemanticInteractionGuidance(
                dim,
                scale=float(hparams.get("cipt_sigm_scale", 1.0)),
                init_std=float(hparams.get("cipt_sigm_init_std", 0.02)),
            )
            # The parent constructs its optimizer before we replace TDA.
            # Rebuild it so SIGM parameters are optimized, not the old TDA.
            trainable = (p for p in self.parameters() if p.requires_grad)
            self.optimizer = get_optimizer(
                hparams["optimizer"], trainable,
                lr=hparams["lr"], weight_decay=hparams["weight_decay"],
            )
            self.trainable_parameter_count = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

        def coefficient(name, default):
            value = hparams.get(name)
            value = float(default if value is None else value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(name + " must be finite and non-negative")
            return value

        # E classification is the unit-weight anchor of the new objective.
        # S uniformity and the other regularizers have separate coefficients;
        # do not inherit the source branch's duplicated E CE via 1+beta.
        self.e_cls_weight = coefficient("cipt_e_cls_weight", 1.0)
        configured_s_weight = coefficient("cipt_s_uniform_weight", self.beta)
        self.s_uniform_weight = configured_s_weight if self.use_de else 0.0
        self.env_kl_weight = coefficient("cipt_env_kl_weight", 0.1)
        self.fused_cls_weight = coefficient("cipt_fused_cls_weight", 0.0)
        self.env_kl_active = self.use_tda and self.env_kl_weight > 0.0
        self.fused_cls_active = self.use_tda and self.fused_cls_weight > 0.0
        self.env_active = self.env_kl_active or self.fused_cls_active

        if self.env_active or self.inference_mode != "e_only":
            if self.cipt_template_mode not in ("b5a", "b5c"):
                raise ValueError("env_kl requires class-agnostic b5a or b5c prompts")
            bank_size = self.text_features.full_intervention_features().shape[0]
            if self.env_kl_active and bank_size < 2:
                raise ValueError("env_kl requires at least two environment prompts")
            if (self.env_kl_active and self.prompt_selector_mode != "all"
                    and int(hparams["cipt_k"]) < 2):
                raise ValueError("env_kl requires cipt_k >= 2")

        print(
            "CIPTDCCL env_kl: fusion_mode={}, inference_mode={}, "
            "SIGM_trainable={}, E_CE_weight={}, S_uniform_weight={}, "
            "IND_weight={}, prompt_KL_weight={}, prompt_KL_active={}, "
            "fused_CE_weight={}, fused_CE_active={}, "
            "KL=ordered_pair_mean, detach=False, "
            "KL_warmup=0".format(
                self.fusion_mode, self.inference_mode,
                int(self.fusion_mode == "sigm"),
                self.e_cls_weight, self.s_uniform_weight,
                self.gamma if self.use_ind else 0.0,
                self.env_kl_weight, self.env_kl_active,
                self.fused_cls_weight, self.fused_cls_active,
            )
        )

    def _clip_ce_visual(self, images):
        # CLIP is frozen; leave gradient flow through the text encoder only.
        with torch.no_grad():
            return F.normalize(self.clip_model.encode_image(images).float(), dim=-1)

    def _clip_ce_logits(self, images):
        visual = self._clip_ce_visual(images)
        text = self.text_encoder(
            self.prompt_learner(), self.prompt_learner.tokenized_prompts
        )
        text = F.normalize(text.float(), dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        return scale * visual @ text.t()

    def update(self, x, y, **kwargs):
        if self.objective_mode == "clip_ce":
            logits = self._clip_ce_logits(torch.cat(x))
            loss = F.cross_entropy(logits, torch.cat(y))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {
                "total_loss": loss.item(),
                "cipt_base_loss": loss.item(),
                "cipt_cls_loss": loss.item(),
                "clip_ce_loss": loss.item(),
            }
        if self.objective_mode == "legacy":
            return super().update(x, y, **kwargs)

        visual = self._visual(torch.cat(x))
        labels = torch.cat(y)
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        causal_logits = self._logits(causal[:, None, :], class_features)[:, 0]
        zero = causal.new_zeros(())

        if self.use_de and self.s_uniform_weight > 0.0:
            spurious_logits = self._logits(spurious[:, None, :], class_features)[:, 0]
            loss_e, loss_s = decomposition_loss_terms(
                causal_logits, spurious_logits, labels
            )
        else:
            loss_e = F.cross_entropy(causal_logits, labels)
            loss_s = zero
        loss_ind = independence_loss(causal, spurious) if self.use_ind else zero
        base = (
            self.e_cls_weight * loss_e
            + self.s_uniform_weight * loss_s
            + self.gamma * loss_ind
        )

        loss_env = zero
        loss_fused_cls = zero
        env_shift = zero
        env_spread = zero
        env_l2_shift = zero
        env_spread_l2 = zero
        env_pred_agreement = zero
        selector_metrics = self._empty_selector_metrics(causal, prompt_count=0)
        if self.env_active:
            # Selection is label-free. Gathered features retain the E/TDA graph;
            # share one fusion pass for CE and MCC, keeping each prompt separate.
            interventions, selector_metrics = self._select_interventions(
                visual, causal, class_features, labels=None
            )
            prompt_logits = self._logits(interventions, class_features)
            if self.fused_cls_active:
                # Each environment predicts the true class using the same
                # learnable class-text anchors. Average CE, not fused features.
                loss_fused_cls = cipt_classification_loss(prompt_logits, labels)
            if self.env_kl_active:
                loss_env = prompt_consistency_loss(prompt_logits)
            with torch.no_grad():
                unit_z = F.normalize(interventions.float(), dim=-1)
                unit_e = F.normalize(causal.float(), dim=-1)[:, None, :]
                env_shift = (1.0 - (unit_z * unit_e).sum(dim=-1)).mean()
                # Zero when all prompts produce the same normalized feature.
                env_spread = (
                    unit_z - unit_z.mean(dim=1, keepdim=True)
                ).square().sum(dim=-1).mean()
                # Raw-space diagnostics complement the normalized measurements.
                raw_z = interventions.float()
                env_l2_shift = (raw_z - causal.float()[:, None, :]).norm(dim=-1).mean()
                env_spread_l2 = (
                    raw_z - raw_z.mean(dim=1, keepdim=True)
                ).square().sum(dim=-1).mean()
                env_pred_agreement = (
                    prompt_logits.argmax(dim=-1)
                    == causal_logits.argmax(dim=-1)[:, None]
                ).float().mean()

        loss_con, valid_fraction, con_weight, neighbor_stats = (
            self._causal_contrastive_loss(visual, causal, labels, y)
        )
        env_term = self.env_kl_weight * loss_env
        fused_cls_term = self.fused_cls_weight * loss_fused_cls
        base = base + fused_cls_term
        total = base + con_weight * loss_con + env_term

        if self.debug_shapes:
            print(
                "CIPTDCCL env_kl shapes: E={} S={} class_logits={} "
                "prompt_logits={} fused_CE_active={}".format(
                    tuple(causal.shape), tuple(spurious.shape),
                    tuple(causal_logits.shape),
                    tuple(prompt_logits.shape) if self.env_active else None,
                    self.fused_cls_active,
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            **{name: value.item() for name, value in neighbor_stats.items()},
            **self._selector_metric_items(selector_metrics),
            "total_loss": total.item(),
            "cipt_base_loss": base.item(),
            # Compatibility diagnostics only; DE is NOT added again to total.
            "cipt_cls_loss": loss_e.item(),
            "cipt_de_loss": (loss_e + loss_s).item() if self.use_de else 0.0,
            "cipt_de_orig_loss": (loss_e + loss_s).item() if self.use_de else 0.0,
            "cipt_de_aug_loss": 0.0,
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": loss_env.item(),
            "dccl_contrastive_loss": loss_con.item(),
            "contrastive_weight_eff": float(con_weight),
            "contrastive_valid_anchor_fraction": valid_fraction.item(),
            "pre_cl_loss": 0.0,
            "reg_loss": 0.0,
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(causal, spurious, dim=-1).mean().item(),
            "ablation_de_enabled": float(self.use_de),
            "ablation_ind_enabled": float(self.use_ind),
            "ablation_tda_enabled": float(self.use_tda),
            "ablation_contrastive_enabled": float(self.use_contrastive),
            "e_cls_loss": loss_e.item(),
            "fused_cls_loss": loss_fused_cls.item(),
            "s_uni_loss": loss_s.item(),
            "env_kl_loss": loss_env.item(),
            "e_cls_w": self.e_cls_weight,
            "fused_cls_w": self.fused_cls_weight if self.fused_cls_active else 0.0,
            "s_uni_w": self.s_uniform_weight,
            "env_kl_w": self.env_kl_weight if self.env_kl_active else 0.0,
            "e_cls_term": (self.e_cls_weight * loss_e).item(),
            "fused_cls_term": fused_cls_term.item(),
            "s_uni_term": (self.s_uniform_weight * loss_s).item(),
            "env_kl_term": env_term.item(),
            "env_shift": env_shift.item(),
            "env_spread": env_spread.item(),
            "env_l2_shift": env_l2_shift.item(),
            "env_spread_l2": env_spread_l2.item(),
            "env_pred_agreement": env_pred_agreement.item(),
            "sigm_active": float(self.fusion_mode == "sigm"),
            "inference_fused": float(self.inference_mode != "e_only"),
            "env_active": float(self.env_active),
            "env_kl_active": float(self.env_kl_active),
            "fused_cls_active": float(self.fused_cls_active),
            "e_only_infer": float(self.inference_mode == "e_only"),
        }

    def predict(self, x):
        if self.objective_mode == "clip_ce":
            return self._clip_ce_logits(x)
        if self.objective_mode == "legacy":
            return super().predict(x)
        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        if self.inference_mode == "e_only":
            # Original env_kl deployment: no text selection/fusion at test time.
            return self._logits(causal[:, None, :], class_features)[:, 0]

        # Inference ablations are explicitly opt-in. Never use target labels.
        interventions, _ = self._select_interventions(
            visual, causal, class_features, labels=None
        )
        prompt_logits = self._logits(interventions, class_features)
        if self.inference_mode == "mean_logits":
            return prompt_logits.mean(dim=1)
        # Log probabilities produce the same argmax as mean softmax probabilities.
        # logsumexp avoids underflow with CLIP's high logit scale.
        return (
            torch.logsumexp(F.log_softmax(prompt_logits, dim=-1), dim=1)
            - math.log(prompt_logits.shape[1])
        )

    def predict_embed(self, x):
        if self.objective_mode == "clip_ce":
            return self._clip_ce_visual(x)
        return super().predict_embed(x)


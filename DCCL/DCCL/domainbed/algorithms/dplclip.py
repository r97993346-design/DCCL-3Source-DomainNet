"""Domain Prompt Learning with a frozen OpenAI CLIP ViT-B/16."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from domainbed import networks
from domainbed.algorithms.algorithms import Algorithm
from domainbed.lib import dplclip_protocol


class DPLCLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        import clip

        path = hparams["dpl_clip_path"] or hparams["clip_backbone"]
        model, preprocess = clip.load(path, device="cpu", jit=False)
        self.clip_model = model.float().requires_grad_(False).eval()
        if hparams["dpl_smoke_audit"]:
            from PIL import Image
            from domainbed.datasets.transforms import clip_basic

            sample = Image.effect_noise((301, 229), 45).convert("RGB")
            torch.testing.assert_close(clip_basic(sample), preprocess(sample), rtol=0, atol=0)
        width = self.clip_model.text_projection.shape[1]
        self.prompt_length = hparams["num_domain_tokens"]
        prefix = " ".join(["X"] * self.prompt_length)
        prompts = [f"{prefix} {name.replace('_', ' ')}." for name in hparams["class_names"]]
        tokens = clip.tokenize(prompts)
        with torch.no_grad():
            embeddings = self.clip_model.token_embedding(tokens)
        self.register_buffer("token_prefix", embeddings[:, :1].contiguous(), persistent=False)
        self.register_buffer("token_suffix", embeddings[:, 1 + self.prompt_length:].contiguous(),
                             persistent=False)
        self.register_buffer("eot_positions", tokens.argmax(dim=-1), persistent=False)

        self.prompt_generator = networks.MLP(width, width * self.prompt_length, hparams)
        for layer in self.prompt_generator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        self.optimizer = torch.optim.SGD(
            self.prompt_generator.parameters(), lr=hparams["lr"],
            momentum=hparams["momentum"], weight_decay=hparams["weight_decay"],
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def text_features(self, image_features):
        context = self.prompt_generator(image_features).mean(dim=0)
        context = context.reshape(self.prompt_length, -1).expand(self.num_classes, -1, -1)
        tokens = torch.cat((self.token_prefix, context, self.token_suffix), dim=1)
        # The text tower stays differentiable with respect to the context.
        model = self.clip_model
        hidden = (tokens + model.positional_embedding).permute(1, 0, 2)
        hidden = model.ln_final(model.transformer(hidden).permute(1, 0, 2))
        rows = torch.arange(self.num_classes, device=hidden.device)
        return hidden[rows, self.eot_positions] @ model.text_projection

    def _logits(self, image_features):
        text_features = self.text_features(image_features)
        return (self.clip_model.logit_scale.exp() * F.normalize(image_features, dim=-1)
                @ F.normalize(text_features, dim=-1).t())

    @torch.no_grad()
    def encode_images(self, images):
        return self.clip_model.encode_image(images)

    @torch.no_grad()
    def predict(self, x):
        return self._logits(self.encode_images(x))

    def update(self, x, y, step=None, **kwargs):
        if len(x) != self.num_domains or len(y) != self.num_domains:
            raise ValueError("DPLCLIP requires one minibatch per source domain")
        audit = self.hparams["dpl_smoke_audit"] and step == 0
        if audit:
            frozen_before = dplclip_protocol.module_digest(self.clip_model)
            generator_before = dplclip_protocol.module_digest(self.prompt_generator)
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for images, labels in zip(x, y):
            # Eq. (6) averages K-way CE across domains. Backpropagate separately
            # to release each differentiable text tower before processing the next.
            loss = F.cross_entropy(self._logits(self.encode_images(images)), labels)
            (loss / self.num_domains).backward()
            losses.append(loss.detach())
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite DPLCLIP loss at step {step}")
        self.optimizer.step()
        result = {"loss": mean_loss, "lr": self.optimizer.param_groups[0]["lr"]}
        if audit:
            grads = [p.grad for p in self.prompt_generator.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            grad_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()
            assert grad_norm > 0
            assert generator_before != dplclip_protocol.module_digest(self.prompt_generator)
            assert all(not p.requires_grad and p.grad is None for p in self.clip_model.parameters())
            assert frozen_before == dplclip_protocol.module_digest(self.clip_model)
            result["smoke_gradient_norm"] = grad_norm
            print(f"DPL smoke passed: loss={mean_loss}, gradient norm={grad_norm}, "
                  f"CLIP state SHA256={frozen_before}", flush=True)
        return result

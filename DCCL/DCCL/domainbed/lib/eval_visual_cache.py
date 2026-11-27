"""Fold-local CPU storage for raw frozen CLIP evaluation batches."""

import torch


class EvalVisualCache:
    def __init__(self):
        self._splits = {}
        self._encoder_state = None
        self.hit_splits = 0

    def prepare(self, algorithm):
        if torch.is_autocast_enabled():
            raise ValueError("Evaluation visual caching requires full-precision evaluation")
        self.hit_splits = 0
        visual = algorithm.clip_model.visual
        parameters = tuple(visual.parameters())
        optimized = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
        if any(p.requires_grad or id(p) in optimized for p in parameters):
            raise ValueError("Evaluation caching requires a frozen, unoptimized CLIP visual encoder")
        if any(module.training for module in visual.modules()):
            raise ValueError("Evaluation caching requires CLIP visual modules in eval mode")

        tensors = parameters + tuple(visual.buffers())
        # Rebuild if the underlying model, weights, precision, or device changes.
        state = (
            id(visual),
            tuple((id(t), t._version, t.dtype, t.device) for t in tensors),
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.allow_tf32,
        )
        if state != self._encoder_state:
            self._splits.clear()
            self._encoder_state = state

    def batches(self, split_name, make_loader, encode, device):
        """Replay the original batch boundaries, order and labels."""
        if split_name in self._splits:
            self.hit_splits += 1
            yield from self._splits[split_name]
            return

        batches = []
        for batch in make_loader():
            with torch.no_grad():
                raw = encode(batch["x"].to(device))
            batches.append({"x": raw.to(device="cpu", copy=True), "y": batch["y"]})
            yield {"x": raw, "y": batch["y"]}
        # A failed/partial evaluation must never publish a partial split.
        self._splits[split_name] = batches

    @property
    def size_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for split in self._splits.values()
            for batch in split
            for tensor in batch.values()
        )

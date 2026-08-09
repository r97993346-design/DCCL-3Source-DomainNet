# Burrowed from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
# modified for the DomainBed.
import copy
import torch
from torch.nn import Module
from copy import deepcopy


def _is_bridge_bn(name, module):
    return (
        "bridge_adapter" in name
        and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    )


def _copy_bridge_bn_buffers(target_model, source_model):
    """Keep SWA Bridge BN buffers current while parameters are averaged.

    SWAD averages parameters only. For Bridge, stale BN buffers can distort the
    checkpoint/segment validation used by LossValley before the final explicit
    BN recalibration. Copying the latest training-model Bridge BN buffers keeps
    segment evaluation coherent without touching frozen ResNet BN buffers.
    """
    source_modules = dict(source_model.named_modules())
    with torch.no_grad():
        for name, target_module in target_model.named_modules():
            if not _is_bridge_bn(name, target_module):
                continue
            source_module = source_modules.get(name)
            if source_module is None or not _is_bridge_bn(name, source_module):
                continue
            if target_module.running_mean is not None:
                target_module.running_mean.copy_(
                    source_module.running_mean.to(target_module.running_mean.device)
                )
            if target_module.running_var is not None:
                target_module.running_var.copy_(
                    source_module.running_var.to(target_module.running_var.device)
                )
            if (
                target_module.num_batches_tracked is not None
                and source_module.num_batches_tracked is not None
            ):
                target_module.num_batches_tracked.copy_(
                    source_module.num_batches_tracked.to(
                        target_module.num_batches_tracked.device
                    )
                )


class AveragedModel(Module):

    def filter(self, model):
        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module

        if hasattr(model, "get_forward_model"):
            model = model.get_forward_model()
            # URERM models use URNetwork, which manages features internally.
            for m in model.modules():
                if hasattr(m, "clear_features"):
                    m.clear_features()

        return model

    def __init__(self, model, device=None, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        self.start_step = -1
        self.end_step = -1
        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module
        model = self.filter(model)
        self.module = deepcopy(model)
        self.module.zero_grad(set_to_none=True)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)

        if device is not None:
            self.module = self.module.to(device)

        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))

        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
                    num_averaged + 1
                )

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        #  return self.predict(*args, **kwargs)
        return self.module(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def predict_embed(self, *args, **kwargs):
        return self.module.predict_embed(*args, **kwargs)

    @property
    def network(self):
        return self.module.network

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        """Update averaged model parameters

        Args:
            model: current model to update params
            step: current step. step is saved for log the averaged range
            start_step: set start_step only for first update
            end_step: set end_step
        """
        model = self.filter(model)
        if isinstance(model, AveragedModel):
            model = model.module
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device))
                )

        # Bridge BN buffers are not parameters and therefore are not handled by
        # the loop above. Keep the segment/final candidate synchronized with the
        # latest source model for reliable SWAD valley selection. Final SWAD is
        # still explicitly recalibrated from training data by update_bn().
        _copy_bridge_bn_buffers(self.module, model)

        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step

        if end_step is not None:
            self.end_step = end_step

    def clone(self):
        clone = copy.deepcopy(self.module)
        clone.optimizer = clone.new_optimizer(clone.network.parameters())
        return clone


def cvt_dbiterator_to_loader(dbiterator, n_iter):
    """Convert DB iterator to the loader"""
    for _ in range(n_iter):
        minibatches = [(x, y) for x, y in next(dbiterator)]
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        yield all_x, all_y


def _batchnorm_targets(model):
    """Choose BN modules whose running statistics should be recomputed.

    DCCLBridgeOfficial keeps ImageNet ResNet BN frozen and introduces BN only in
    the Bridge expectation estimators. If such Bridge BN modules are present,
    refresh only those modules. Other algorithms retain the historical behavior
    of refreshing all BatchNorm modules.
    """
    batch_norms = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    bridge_batch_norms = [
        (name, module)
        for name, module in batch_norms
        if "bridge_adapter" in name
    ]
    return bridge_batch_norms if bridge_batch_norms else batch_norms


@torch.no_grad()
def update_bn(iterator, model, n_steps, device="cuda"):
    """Recompute BN running statistics for the final SWAD model.

    When the model contains ``bridge_adapter`` BatchNorm layers, only those
    Bridge BNs are reset and updated. The pretrained ResNet BatchNorm buffers
    are left untouched. During the refresh the whole model is put in eval mode
    and only the selected BN layers are switched to train mode, which also keeps
    dropout and unrelated stateful layers deterministic.
    """
    targets = _batchnorm_targets(model)
    if not targets:
        return

    momenta = {}
    for _name, module in targets:
        module.running_mean.zero_()
        module.running_var.fill_(1)
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()
        momenta[module] = module.momentum

    was_training = model.training
    model.eval()
    for _name, module in targets:
        module.train()
        module.momentum = None

    for _ in range(n_steps):
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(iterator)
        x = torch.cat([dic["x"] for dic in batches_dictlist])
        x = x.to(device)
        model(x)

    for module, momentum in momenta.items():
        module.momentum = momentum

    model.train(was_training)

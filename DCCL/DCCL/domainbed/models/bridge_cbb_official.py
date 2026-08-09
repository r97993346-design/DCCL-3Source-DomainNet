"""
Bridge CBB adapted from the official implementation:

Repository: MingboHong/Domain-Generlization-Bridge
Commit: 88946a9793e61016f65f4f99ee30e326ae992c54
Source: mmdet/models/backbones/cim_utils.py

The basis projection, expectation estimation, mediator path, fusion order, and
initialization follow the official implementation. A small local replacement
for ``mmcv.cnn.ConvModule`` removes the MMDetection/MMCV runtime dependency.

Normalization intentionally follows the official mixed design: the outer
mediator/fusion convolutions use GroupNorm, while the two expectation-estimator
refine convolutions use BatchNorm. The Bridge BatchNorm statistics are refreshed
selectively on the final SWAD model; frozen ResNet BatchNorm buffers are never
reset by that refresh.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Compatibility subset of mmcv.cnn.ConvModule used by official Bridge.

    This implementation preserves the relevant default order (conv -> norm ->
    activation) and ``bias='auto'`` behavior for the configurations used by
    ``ExpectationEstimator`` and ``MultiScaleBasisBlock``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        bias = norm_cfg is None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = self._build_norm(norm_cfg, out_channels)
        self.activate = self._build_activation(act_cfg)

    @staticmethod
    def _build_norm(norm_cfg, num_features):
        if norm_cfg is None:
            return None
        norm_type = norm_cfg.get("type")
        if norm_type == "BN":
            return nn.BatchNorm2d(num_features)
        if norm_type == "GN":
            num_groups = int(norm_cfg.get("num_groups", 32))
            if num_features % num_groups != 0:
                raise ValueError(
                    f"GroupNorm requires channels divisible by num_groups, got "
                    f"channels={num_features}, num_groups={num_groups}"
                )
            return nn.GroupNorm(num_groups, num_features)
        raise ValueError(f"Unsupported norm_cfg for Bridge ConvModule: {norm_cfg}")

    @staticmethod
    def _build_activation(act_cfg):
        if act_cfg is None:
            return None
        act_type = act_cfg.get("type")
        if act_type == "ReLU":
            return nn.ReLU(inplace=True)
        if act_type == "SiLU":
            return nn.SiLU(inplace=True)
        raise ValueError(f"Unsupported act_cfg for Bridge ConvModule: {act_cfg}")

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


def _build_orthogonal_basis(num_basis, feat_dim):
    basis = torch.randn(num_basis, feat_dim)
    q, _ = torch.linalg.qr(basis.T)
    return nn.Parameter(q.T)


def _get_same_padding(kernel_size):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    return kernel_size // 2


def _project_onto_basis(x, basis, eps=1e-5, normalize_basis=True):
    batch, channels, height, width = x.shape
    assert channels == basis.shape[1]

    basis = basis.to(device=x.device, dtype=x.dtype)
    if normalize_basis:
        basis = F.normalize(basis, p=2, dim=1)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
    gram = basis @ basis.T
    reg = eps * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeffs = torch.linalg.solve(gram + reg, (x_flat @ basis.T).T).T
    x_proj = coeffs @ basis
    return (
        x_proj.view(batch, height, width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _resolve_basis_dim(in_channels, basis_reduction, basis_reduction_mode):
    assert basis_reduction_mode in ["sub", "div", "mul"]
    if basis_reduction_mode == "sub":
        reduced = in_channels - basis_reduction
    elif basis_reduction_mode == "div":
        reduced = in_channels // basis_reduction
    else:
        reduced = int(in_channels * basis_reduction)
    reduced = int(reduced)
    if reduced <= 0 or reduced > in_channels:
        raise ValueError(
            f"Invalid reduced basis dimension {reduced} for "
            f"in_channels={in_channels}, basis_reduction={basis_reduction}, "
            f"mode={basis_reduction_mode}"
        )
    return reduced


class ExpectationEstimator(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        num_basis=128,
        num_samples=128,
        eps=1e-5,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        with_query=True,
        with_ssp=True,
        basis_normalize=True,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.eps = eps
        self.with_query = with_query
        self.with_ssp = with_ssp
        self.basis_normalize = basis_normalize
        conv_padding = _get_same_padding(conv_kernel_size)

        self.sample_query_proj = ConvModule(
            feat_dim,
            num_samples,
            kernel_size=1,
            stride=1,
            norm_cfg=None,
            act_cfg=None,
        )
        self.expectation_basis = _build_orthogonal_basis(num_basis, feat_dim)
        self.refine_conv = ConvModule(
            feat_dim,
            feat_dim,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        if self.with_query:
            sample_response_map = self.sample_query_proj(x)
            sample_attention = F.softmax(sample_response_map, dim=1)
            expected_feature_map = (
                sample_response_map * sample_attention
            ).sum(dim=1, keepdim=True) * x
        else:
            expected_feature_map = x

        if self.with_ssp:
            expected_feature_map = _project_onto_basis(
                expected_feature_map,
                self.expectation_basis,
                eps=self.eps,
                normalize_basis=self.basis_normalize,
            )

        return self.refine_conv(expected_feature_map)


class MultiScaleBasisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        basis_reduction=2,
        basis_reduction_mode="div",
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=dict(type="SiLU"),
        with_ssp=True,
        with_query=True,
        with_input_subspace=False,
        with_dropout=False,
        basis_normalize=True,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.with_input_subspace = with_input_subspace
        self.with_dropout = with_dropout
        self.basis_normalize = basis_normalize
        self.num_reduced_basis = _resolve_basis_dim(
            in_channels, basis_reduction, basis_reduction_mode
        )
        conv_padding = _get_same_padding(conv_kernel_size)

        if self.with_input_subspace:
            self.input_basis = _build_orthogonal_basis(
                self.num_reduced_basis, in_channels
            )

        # Official CBB: mediator/fusion use the block-level GN configuration.
        self.mediator_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Official CBB: estimator refine convolutions keep their own default BN.
        # Do not pass the outer GN configuration into these estimators.
        self.expected_input_estimator = ExpectationEstimator(
            feat_dim=in_channels,
            num_basis=self.num_reduced_basis,
            num_samples=in_channels,
            with_ssp=with_ssp,
            with_query=with_query,
            basis_normalize=basis_normalize,
            conv_kernel_size=conv_kernel_size,
        )
        self.expected_mediator_estimator = ExpectationEstimator(
            feat_dim=in_channels,
            num_basis=self.num_reduced_basis,
            num_samples=in_channels,
            with_ssp=with_ssp,
            with_query=with_query,
            basis_normalize=basis_normalize,
            conv_kernel_size=conv_kernel_size,
        )
        self.fusion_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, ConvModule):
                conv = module.conv
                if isinstance(conv, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        conv.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        mediator_feature = self.mediator_conv(x)
        expected_input_feature = self.expected_input_estimator(x)
        expected_mediator_feature = self.expected_mediator_estimator(
            mediator_feature
        )
        if self.with_input_subspace:
            mediator_feature = _project_onto_basis(
                x,
                self.input_basis,
                normalize_basis=self.basis_normalize,
            )

        fused_feature = (
            mediator_feature
            + expected_input_feature
            + expected_mediator_feature
        )
        return self.fusion_conv(fused_feature)


class ResidualBridgeBlock(nn.Module):
    """Apply CBB through a stable fixed-scale residual adapter.

    The expansion layer is zero-initialized, so the adapter starts as an exact
    identity without a learnable scalar gate. This preserves the pretrained
    feature distribution at step 0 while avoiding the gate=0 gradient bottleneck
    and the extra gate-times-weights interaction under SWAD parameter averaging.
    """

    def __init__(
        self,
        in_channels,
        bridge_channels=256,
        residual_scale=0.1,
        **bridge_kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.bridge_channels = int(bridge_channels)
        self.residual_scale = float(residual_scale)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.bridge_channels <= 0:
            raise ValueError(
                f"bridge_channels must be positive, got {self.bridge_channels}"
            )
        if self.residual_scale < 0.0:
            raise ValueError(
                f"residual_scale must be non-negative, got {self.residual_scale}"
            )

        self.reduce = nn.Conv2d(
            self.in_channels, self.bridge_channels, kernel_size=1, bias=False
        )
        self.bridge_block = MultiScaleBasisBlock(
            in_channels=self.bridge_channels, **bridge_kwargs
        )
        self.expand = nn.Conv2d(
            self.bridge_channels, self.in_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.expand.weight)

    def forward(self, x):
        bridge_delta = self.expand(self.bridge_block(self.reduce(x)))
        return x + self.residual_scale * bridge_delta

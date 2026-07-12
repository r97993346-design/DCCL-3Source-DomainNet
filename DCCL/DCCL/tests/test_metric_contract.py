import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from domainbed.trainer import _mean_numeric_metric
from train_all import str2bool


def test_mean_numeric_metric_accepts_numeric_scalars():
    value = _mean_numeric_metric(
        "loss",
        [1, 2.0, np.float32(3.0), torch.tensor(4.0), True],
    )
    assert value == pytest.approx(2.2)


def test_mean_numeric_metric_rejects_string_with_metric_key():
    with pytest.raises(TypeError, match="Metric 'param_group_lrs'.*str"):
        _mean_numeric_metric("param_group_lrs", ["5e-05,0.0005"])


def test_mean_numeric_metric_rejects_multi_element_tensor():
    with pytest.raises(TypeError, match="Metric 'bad_tensor'.*tensor shape"):
        _mean_numeric_metric("bad_tensor", [torch.ones(2)])


def test_mean_numeric_metric_rejects_nan_and_inf():
    with pytest.raises(FloatingPointError, match="Metric 'nan_metric'"):
        _mean_numeric_metric("nan_metric", [math.nan])
    with pytest.raises(FloatingPointError, match="Metric 'inf_metric'"):
        _mean_numeric_metric("inf_metric", [math.inf])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("false", False), ("False", False), ("0", False), ("true", True), ("True", True), ("1", True)],
)
def test_str2bool_parses_cli_forms(raw, expected):
    assert str2bool(raw) is expected

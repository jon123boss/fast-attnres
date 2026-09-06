"""Regressions for information lost by an uncentered FP32 log-sum-exp."""
import pytest
import torch


@pytest.mark.parametrize('scale', [2**24, -(2**24)])
def test_separate_softmax_statistics_preserve_large_common_offsets(scale):
    keys = torch.ones(4, 2, 16, dtype=torch.bfloat16).float()
    logits = keys.sum(-1) * torch.rsqrt(keys.square().mean(-1) + 2**-23) * scale
    maximum = logits.max(0).values
    numerator = (logits - maximum).exp()
    denominator = numerator.sum(0)
    lossy = (logits - (maximum + denominator.log())).exp()
    assert torch.equal(lossy, torch.ones_like(lossy))
    torch.testing.assert_close(numerator / denominator, logits.softmax(0), rtol=0, atol=0)


def test_fake_softmax_statistics_shape_covers_both_checkpoint_policies():
    from attnres._kernels.fixed_tail_sources import _source_forward_fake
    for count in (2, 4):
        result = _source_forward_fake([torch.empty(2, 256)] * count, torch.empty(16),
                                      2**-23, 1.0)
        assert len(result) == 5
        assert result[4].shape == (2, 2)
        assert result[4].dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_large_equal_logits_and_changed_upstream_graph_replay():
    from attnres import attnres
    from validation.softmax_checks import run_equal_logit_checks
    assert len(run_equal_logit_checks(attnres)) == 8

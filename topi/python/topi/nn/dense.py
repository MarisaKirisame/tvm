"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag

def dense_default(data, weight, bias=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        Tensor with shape [batch0, batch1, batch2..., in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        Tensor with shape [batch0, batch1, batch2..., out_dim]
    """
    assert len(data.shape) > 0 and len(weight.shape) == 2, \
        "bad shape"
    if bias is not None:
        assert len(bias.shape) == 1
    batch = data.shape[:-1]
    in_dim = data.shape[-1]
    out_dim = weight.shape[0]
    k = tvm.reduce_axis((0, in_dim), name='k')
    out_shape = batch + [out_dim]
    def matmul_func(*idx):
        batch_idx = idx[:-1]
        out_idx = idx[-1]
        return tvm.sum(data[tuple(batch_idx + (k,))] * weight[out_idx, k], axis=k)
    matmul = tvm.compute(out_shape, matmul_func, tag='dense')
    if bias is not None:
        def matmul_func(*idx):
            batch_idx = idx[:-1]
            out_idx = idx[-1]
            return matmul[tuple(batch_idx + (k,))] + bias[out_idx]
        matmul = tvm.compute(out_shape, matmul_func, tag=tag.BROADCAST)
    return matmul


@tvm.target.override_native_generic_func("dense")
def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    return dense_default(data, weight, bias)

import numpy as np
import tvm
from tvm import relay
from tvm.relay.ir_pass import partial_eval, alpha_equal, infer_type, dead_code_elimination, gradient
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def dcpe(expr):
    return dead_code_elimination(partial_eval(expr))


def test_tuple():
    t = relay.TypeVar("t")
    x = relay.Var("x", t)
    body = relay.TupleGetItem(relay.Tuple([relay.const(4.0), x]), 1)
    f = relay.Function([x], body, None, [t])
    assert alpha_equal(dcpe(f), relay.Function([x], x, None, [t]))


def test_const_inline():
    d = relay.Var("d")
    double = relay.Function([d], d + d)
    orig = double(relay.const(4.0))
    assert alpha_equal(dcpe(double(relay.const(4.0))), relay.const(8.0))


def test_ref():
    d = relay.Var("d")
    r = relay.Var("r")
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = relay.Let(x, relay.RefWrite(r, relay.RefRead(r) * relay.RefRead(r)), body)
    body = relay.Let(r, relay.RefCreate(d), body)
    square = relay.Function([d], body)
    assert alpha_equal(dcpe(square), relay.Function([d], d * d))


def test_ad():
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    d = relay.Var("d", t)
    f = relay.Function([d], d * d)
    g = dcpe(gradient(f))
    m = d * d
    o = relay.op.ones_like(m)
    grad = relay.op.zeros_like(d) + relay.op.collapse_sum_like(o * d, d) + relay.op.collapse_sum_like(o * d, d)
    expected = relay.Function([d], relay.Tuple([m, relay.Tuple([grad])]))
    assert alpha_equal(g, expected)


if __name__ == '__main__':
    test_tuple()
    test_const_inline()
    test_ref()
    test_ad()

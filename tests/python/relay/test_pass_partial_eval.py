import numpy as np
import tvm
from tvm import relay
from tvm.relay.ir_pass import partial_eval, alpha_equal, infer_type, dead_code_elimination, gradient
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude
from tvm.relay import create_executor


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


def test_if_ref():
    shape = ()
    dtype = 'bool'
    t = relay.TensorType(shape, dtype)
    d = relay.Var("d", t)
    r = relay.Var("r")
    update = relay.Function([], relay.RefWrite(r, relay.RefRead(r) + relay.RefRead(r)))
    u = relay.Var("u")
    body = relay.If(d, u(), u())
    eff = relay.Var("eff")
    body = relay.Let(eff, body, relay.RefRead(r))
    f = relay.Function([d], relay.Let(r, relay.RefCreate(relay.const(1)), relay.Let(u, update, body)))
    f = infer_type(f)
    pe_f = infer_type(partial_eval(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(relay.const(True))
    pe_f_res = ex.evaluate(pe_f)(relay.const(True))
    np.testing.assert_allclose(f_res.asnumpy(), 2 * np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), 2 * np.ones_like(pe_f_res.asnumpy()))

if __name__ == '__main__':
    test_tuple()
    test_const_inline()
    test_ref()
    test_ad()
    test_if_ref()

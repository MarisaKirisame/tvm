import numpy as np
import tvm
from tvm import relay
from tvm.relay.ir_pass import partial_eval, alpha_equal, infer_type, dead_code_elimination, gradient
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude
from tvm.relay import create_executor
from tvm.relay import Var, TypeVar, TupleGetItem, Let, Function, const, RefRead, RefWrite, RefCreate, TensorType, Tuple, If, Module, Clause, PatternConstructor, PatternVar, Match

def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def dcpe(expr):
    return dead_code_elimination(partial_eval(expr))


def test_tuple():
    t = TypeVar("t")
    x = Var("x", t)
    body = TupleGetItem(relay.Tuple([relay.const(4.0), x]), 1)
    f = Function([x], body, None, [t])
    assert alpha_equal(dcpe(f), relay.Function([x], x, None, [t]))


def test_const_inline():
    d = Var("d")
    double = Function([d], d + d)
    orig = double(const(4.0))
    assert alpha_equal(dcpe(double(const(4.0))), const(8.0))


def test_ref():
    d = relay.Var("d")
    r = relay.Var("r")
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = Let(x, RefWrite(r, RefRead(r) * RefRead(r)), body)
    body = Let(r, RefCreate(d), body)
    square = Function([d], body)
    assert alpha_equal(dcpe(square), Function([d], d * d))


def test_ad():
    shape = (10, 10)
    dtype = "float32"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    f = Function([d], d * d)
    g = dcpe(gradient(f))
    m = d * d
    x = relay.Var("x")
    o = op.ones_like(x)
    x1 = relay.Var("x1")
    grad = op.zeros_like(d) + op.collapse_sum_like(x1 * d, d) + op.collapse_sum_like(x1 * d, d)
    body = Tuple([x, Tuple([grad])])
    body = relay.Let(x1, o, body)
    expected = Function([d], relay.Let(x, m, body))
    assert alpha_equal(g, expected)


def test_if_ref():
    shape = ()
    dtype = "bool"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    r = Var("r")
    update = Function([], RefWrite(r, RefRead(r) + RefRead(r)))
    u = Var("u")
    body = If(d, u(), u())
    eff = Var("eff")
    body = Let(eff, body, RefRead(r))
    f = Function([d], Let(r, RefCreate(const(1)), Let(u, update, body)))
    f = infer_type(f)
    pe_f = infer_type(partial_eval(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(const(True))
    pe_f_res = ex.evaluate(pe_f)(const(True))
    np.testing.assert_allclose(f_res.asnumpy(), 2 * np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), 2 * np.ones_like(pe_f_res.asnumpy()))


def test_function_invalidate():
    shape = ()
    dtype = "bool"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    r = Var("r")
    fetch = Function([], RefRead(r))
    fet = Var("fetch")
    fet_obscured = Var("fetch_obscured")
    u = Var("u")
    body = If(d, fet_obscured(), fet_obscured())
    body = Let(u, RefWrite(r, const(1)), body)
    body = Let(fet_obscured, If(d, fet, fet), body)
    body = Let(fet, fetch, body)
    body = Let(r, RefCreate(const(0)), body)
    f = Function([d], body)
    f = infer_type(f)
    pe_f = infer_type(partial_eval(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(const(True))
    pe_f_res = ex.evaluate(pe_f)(const(True))
    np.testing.assert_allclose(f_res.asnumpy(), np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), np.ones_like(pe_f_res.asnumpy()))


def test_head_cons():
    mod = Module()
    p = Prelude(mod)
    def hd_impl():
        a = TypeVar("a")
        x = Var("x", p.l(a))
        y = Var("y")
        z = Var("z")
        cons_case = Clause(PatternConstructor(p.cons,
                                                          [PatternVar(y),
                                                           PatternVar(z)]),
                                 y)
        return Function([x], Match(x, [cons_case]), a, [a])
    t = TypeVar("t")
    x = Var("x", t)
    hd = Var("hd")
    body = Let(hd, hd_impl(), hd(p.cons(x, p.nil())))
    f = Function([x], body, None, [t])
    f = infer_type(f, mod=mod)
    res = dcpe(f)
    assert alpha_equal(res, Function([x], x, t, [t]))


def test_map():
    mod = Module()
    p = Prelude(mod)
    f = Var("f")
    orig = p.map(p.cons(const(1), p.cons(const(2), p.cons(const(3), p.nil()))))
    print(dcpe(orig))

if __name__ == '__main__':
    test_tuple()
    test_const_inline()
    test_ref()
    test_ad()
    test_if_ref()
    test_function_invalidate()
    test_head_cons()
    #test_map()

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import absolute_import, print_function

import argparse, json, os, requests, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay.transform import PartialEvaluate, ToANormalForm, DeadCodeElimination

import vta
from vta.testing import simulator
from vta.top import graph_pack, GraphPack
import aot
import network
from network.tlstm import TreeLSTM

# Make sure that TVM was compiled with RPC=1
assert tvm.module.enabled("rpc")

env = vta.get_env()

device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
model = "tree_lstm"
start_pack="nn.max_pool2d"
stop_pack="nn.global_avg_pool2d"

######################################################################
# Obtain an execution remote
# ---------------------------------
# When target is 'pynq', reconfigure FPGA and runtime.
# Otherwise, if target is 'sim', execute locally.
if env.TARGET not in ["sim", "tsim"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
    device_host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, int(tracker_port), timeout=10000)

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

from tvm.relay import ExprMutator, ExprVisitor

class LetList:
    VAR_COUNTER = 0

    def __init__(self):
        self.bindings = []

    def push(self, expr, *, ty=None, bind_var=None):
        if bind_var is None:
            bind_var = relay.Var(f'fresh{LetList.VAR_COUNTER}', ty)
            LetList.VAR_COUNTER += 1
        self.bindings.append((bind_var, expr))
        return bind_var

    def get(self, expr):
        ret = expr
        for (var, rhs) in reversed(self.bindings):
            ret = relay.Let(var, rhs, ret)
        return ret

    def with_ll(func):
        ll = LetList()
        return ll.get(func(ll))


def get_shape(expr):
    return [int(x) for x in expr.checked_type.shape]


class Config:
    def __init__(self, n=1, i=8):
        self.n = n
        self.i = i
        self.o = i


class Layout:
    def key(self):
        raise NotImplementedError(type(self))

    def keyname(self):
        return (self.key(), type(self).__name__)

    def __eq__(self, other):
        return self.keyname() == other.keyname()

    def __hash__(self):
        return hash(self.keyname())

    def to_layout(self, expr, config, original_shape):
        raise NotImplementedError(type(self))

    def from_layout(self, expr, config, original_shape):
        raise NotImplementedError(type(self))


def full_div(l, r):
    assert l % r == 0
    return l // r


class NIniLayout(Layout):
    def key(self):
        return ()

    def to_layout(self, expr, config, original_shape):
        # NI
        assert len(original_shape) == 2
        N, I = original_shape
        n, i = config.n, config.i
        # NnIi
        expr = relay.op.reshape(expr,
                                newshape=(full_div(N, n), n,
                                          full_div(I, i), i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

    def from_layout(self, expr, config, original_shape):
        assert len(original_shape) == 2
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        expr = relay.op.reshape(expr, newshape=original_shape)
        return expr

class OIoiLayout(Layout):
    def __init__(self, I):
        self.I = I

    def key(self):
        return self.I

    def to_layout(self, expr, config, original_shape):
        # probably packing incorrectly. but operation cost is the same so no worry.
        # NI
        assert len(original_shape) == 2
        I = self.I
        O = full_div(original_shape[0] * original_shape[1], I)
        i, o = config.i, config.o
        # NnIi
        expr = relay.op.reshape(expr,
                                newshape=(full_div(O, o), o,
                                          full_div(I, i), i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

class BiasNIniLayout(Layout):
    def __init__(self, N):
        assert N == 1
        self.N = N

    def key(self):
        return self.N

    def to_layout(self, expr, config, original_shape):
        assert len(original_shape) == 1
        I = original_shape[0]
        i = config.i
        assert I % i == 0
        # NnIi
        expr = relay.op.reshape(expr, newshape=(1, 1, I // i, i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

class IdentityLayout(Layout):
    def key(self):
        return ()

    def to_layout(self, expr, config, original_shape):
        return expr

    def from_layout(self, expr, config, original_shape):
        return expr

def safe_zip(l, r):
    assert len(l) == len(r)
    return zip(l, r)

# NIni input/output, OIoi weight
def layout_vta(expr):
    assert isinstance(expr, relay.Function)

    expr_layouts = dict()
    ops_result_layouts = dict()
    rhs_let = set()
    progress = True

    def add(expr, layout):
        nonlocal progress
        if expr not in expr_layouts:
            expr_layouts[expr] = set()
        if layout not in expr_layouts[expr]:
            expr_layouts[expr].add(layout)
            progress = True

    def add_ops(expr, in_layouts, out_layout):
        ops_result_layouts[expr] = (in_layouts, out_layout)
        add(expr, out_layout)
        for arg, layout in safe_zip(expr.args, in_layouts):
            add(arg, layout)

    class VtaLayout(ExprVisitor):
        def visit_call(self, expr):
            if isinstance(expr.op, relay.Op):
                if expr.op.name == 'nn.dense':
                    x, y = expr.args
                    x_layout = NIniLayout()
                    y_layout = OIoiLayout(get_shape(x)[1])
                    add_ops(expr, (x_layout, y_layout),  NIniLayout())
                elif expr.op.name == 'add' and all([arg in expr_layouts for arg in expr.args]):
                    x, y = expr.args
                    assert len(get_shape(x)) == 2
                    x_layout = NIniLayout()
                    if len(get_shape(y)) == 1:
                        y_layout = BiasNIniLayout(get_shape(x)[0])
                    else:
                        assert len(get_shape(y)) == 2
                        y_layout = NIniLayout()
                    add_ops(expr, (x_layout, y_layout), NIniLayout())

            super().visit_call(expr)

        def visit_let(self, expr):
            if expr.value in expr_layouts:
                for layout in expr_layouts[expr.value]:
                    add(expr.var, layout)
            rhs_let.add(expr.value)
            super().visit_let(expr)

    for param in expr.params:
        if isinstance(param.checked_type, relay.TensorType):
            add(param, IdentityLayout())

    while progress:
        progress = False
        VtaLayout().visit(expr)

    # either variable or a call with variable args
    for expr in expr_layouts:
        if isinstance(expr, relay.Var):
            pass
        elif isinstance(expr, relay.Call):
            assert(expr in rhs_let)
            assert(all([isinstance(arg, relay.Var) and arg in expr_layouts for arg in expr.args]))
        else:
            assert False
    return expr_layouts, ops_result_layouts


def rewrite_vta(expr, expr_layouts, ops_result_layouts, config=None):
    if config is None:
        config = Config()
    class VtaRewrite(ExprMutator):
        def __init__(self):
            self.vta_map = {}
            super().__init__()

        def transform_var(self, var, ll):
            assert isinstance(var, relay.Var)
            if var in expr_layouts:
                for layout in expr_layouts[var]:
                    if (var, layout) not in self.vta_map:
                        self.vta_map[(var, layout)] = ll.push(layout.to_layout(var, config, get_shape(var)))
            return var

        def visit_let(self, expr):
            if isinstance(expr.value, relay.Call) and expr.value in expr_layouts:
                assert isinstance(expr.value.op, relay.Op)
                def _with_func(ll):
                    assert expr.value in ops_result_layouts
                    _, layout = ops_result_layouts[expr.value]
                    vta_var = ll.push(self.transform(expr.value))
                    self.vta_map[(expr.value, layout)] = vta_var
                    self.vta_map[(expr.var, layout)] = vta_var
                    ll.push(layout.from_layout(vta_var, config, get_shape(expr.value)), bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            elif isinstance(expr.value, relay.Var) and expr.value in expr_layouts:
                def _with_func(ll):
                    for layout in expr_layouts[expr.value]:
                        self.vta_map[(expr.var, layout)] = self.vta_map[(expr.value, layout)]
                    ll.push(expr.value, bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            elif expr.var in expr_layouts:
                def _with_func(ll):
                    ll.push(self.visit(expr.value), bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            else:
                return super().visit_let(expr)

        def transform_clause(self, clause):
            def _with_func(ll):
                for var in relay.analysis.bound_vars(clause.lhs):
                    self.transform_var(var, ll)
                return self.visit(clause.rhs)
            return relay.Clause(clause.lhs, LetList.with_ll(_with_func))

        def visit_match(self, expr):
            clauses = [self.transform_clause(clauses) for clauses in expr.clauses]
            return relay.Match(self.visit(expr.data), clauses, expr.complete)

        def visit_function(self, expr):
            def _with_func(ll):
                for var in expr.params:
                    self.transform_var(var, ll)
                return self.visit(expr.body)
            return relay.Function(
                    expr.params,
                    LetList.with_ll(_with_func),
                    expr.ret_type,
                    expr.type_params,
                    expr.attrs)

        def get_vta_map(self, expr, layout):
            if (expr, layout) not in self.vta_map:
                print("not find! vta_map:")
                print(self.vta_map)
                print("not find! layout:")
                print((expr, layout))
                for x, y in self.vta_map.keys():
                    if x == expr:
                        print("expr in!")
                        print(y)
                        print(layout)
                        print(y == layout)
            assert (expr, layout) in self.vta_map
            return self.vta_map[(expr, layout)]

        def transform(self, expr):
            assert expr in ops_result_layouts
            in_layouts, _ = ops_result_layouts[expr]
            remapped_args = [self.get_vta_map(arg, layout) for arg, layout in safe_zip(expr.args, in_layouts)]
            new_call = relay.Call(expr.op, remapped_args, expr.attrs, expr.type_args)
            ty = expr.checked_type
            assert isinstance(ty, relay.TensorType)
            shape = [int(x) for x in ty.shape]

            if expr.op.name == 'add':
                return self.transform_add(new_call, shape)
            elif expr.op.name == 'nn.dense':
                return self.transform_dense(new_call, shape)
            else:
                raise

        def transform_add(self, expr, old_shape):
            return expr

        def transform_dense(self, expr, old_shape):
            return relay.op.nn.NCncdense(expr.args[0], expr.args[1])

    return VtaRewrite().visit(expr)


treelstm = TreeLSTM(input_size=128, memory_size=256, dtype="int32")
mod = treelstm.mod
mod = ToANormalForm()(mod)
mod = PartialEvaluate()(mod)
mod = DeadCodeElimination()(mod)
mod["main"] = treelstm.get()

import pprint

print(mod["f_0"])
expr_layouts, ops_result_layouts = layout_vta(mod["f_0"])
rewritten_f0 = rewrite_vta(mod["f_0"], expr_layouts, ops_result_layouts)
mod["f_0"] = rewritten_f0
rewritten_f0 = mod["f_0"]
print(rewritten_f0)

# Load pre-configured AutoTVM schedules
with autotvm.tophub.context(target):
    def get_f():
        f = aot.compile(mod["main"], mod, ctx, target)

    # Compile Relay program with AlterOpLayout disabled
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        if target.device_name != "vta":
            f = get_f()
        else:
            with vta.build_config():
                f = get_f()

    # Send the inference library over to the remote RPC server
    temp = util.tempdir()
    lib.save(temp.relpath("graphlib.o"))
    remote.upload(temp.relpath("graphlib.o"))
    lib = remote.load_module("graphlib.o")

raise
if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()
    tvm_output = f(image)
    # timer()
    sim_stats = simulator.stats()
    print("\nExecution statistics:")
    for k, v in sim_stats.items():
        # Since we execute the workload many times, we need to normalize stats
        # Note that there is always one warm up run
        # Therefore we divide the overall stats by (num * rep + 1)
        print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
else:
    tcost = timer()
    std = np.std(tcost.results) * 1000
    mean = tcost.mean * 1000
    print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
    print("Average per sample inference time: %.2fms" % (mean/env.BATCH))

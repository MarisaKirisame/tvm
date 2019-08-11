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

from tvm.relay import ExprVisitor

def analyze_vta(expr):
    assert isinstance(expr, relay.Function)

    vta_nodes = set()
    progress = True

    def add(expr):
        nonlocal progress
        if expr not in vta_nodes:
            vta_nodes.add(expr)
            progress = True

    class VtaAnalyze(ExprVisitor):
        def visit_call(self, expr):
            if isinstance(expr.op, relay.Op):
                if expr.op.name == 'nn.dense':
                    add(expr)
                    for arg in expr.args:
                        add(arg)
                elif expr.op.name == 'add' and all([arg in vta_nodes for arg in expr.args]):
                    add(expr)
            super().visit_call(expr)

        def visit_let(self, expr):
            if expr.value in vta_nodes:
                add(expr.var)
            super().visit_let(expr)

    for param in expr.params:
        if isinstance(param.checked_type, relay.TensorType):
            vta_nodes.add(param)
    while progress:
        progress = False
        VtaAnalyze().visit(expr)
    return vta_nodes


treelstm = TreeLSTM(input_size=128, memory_size=256, dtype="int32")
mod = treelstm.mod
mod = ToANormalForm()(mod)
mod = PartialEvaluate()(mod)
mod = DeadCodeElimination()(mod)
mod["main"] = treelstm.get()
import pprint
pprint.pprint(analyze_vta(mod["f_0"]))

raise
# Load pre-configured AutoTVM schedules
with autotvm.tophub.context(target):
    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        mod = GraphPack(env.BATCH,
                        env.BLOCK_OUT,
                        env.WGT_WIDTH,
                        start_pack,
                        stop_pack)(mod)

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

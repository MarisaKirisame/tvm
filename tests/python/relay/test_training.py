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
import collections
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay.analysis import free_vars, free_type_vars
from tvm.relay import create_executor, transform
from tvm.relay.transform import gradient, SimplifyInference
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions, make_nat_expr, run_infer_type, check_grad, rand, count_ops
import tvm.relay.op as op

from tvm.relay.testing.resnet import get_workload

def test_resnet_ad():
    x, y = get_workload(batch_norm=False)
    func = x["main"]
    back_func = gradient(func)
    x["bp"] = back_func

if __name__ == "__main__":
    #pytest.main([__file__])
    test_resnet_ad()

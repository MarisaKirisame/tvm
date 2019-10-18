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
"""Operators for manipulating low-level memory."""
from __future__ import absolute_import as _abs
from . import _make
from .... import nd as _nd
from .... import TVMContext as _TVMContext

def alloc_tensor(storage, shape, dtype='float32', assert_shape=None):
    """Allocate a tensor with the provided shape, and dtype.

    Parameters
    ----------
    storage : tvm.relay.Expr
        The storage to allocate from.

    shape : tvm.relay.Expr
        The shape of the tensor to allocate.

    dtype: str
        The dtype of the tensor.

    assert_shape: TODO

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_tensor expression.
    """
    return _make.alloc_tensor(storage, shape, dtype, assert_shape)

def alloc_storage(size, alignment, dtype_hint='float32'):
    """Annotate an expression to prevent it being fused with previous expressions.

    Parameters
    ----------
    size : tvm.relay.Expr
        The size of the allocation.
    alignment : tvm.relay.Expr
        The alignment of the allocation.
    dtype : str
        The dtype_hint of the allocation.

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_storage expression.
    """
    return _make.alloc_storage(size, alignment, dtype_hint)

def shape_func(func, inputs, outputs, dependent=False):
    """Annotate an expression to prevent it being fused with previous expressions.

    Parameters
    ----------
    func : tvm.relay.Expr
        The primitive function from which to compute the shape function.
    inputs : tvm.relay.Tuple
        The tupled inputs.
    outputs : tvm.relay.Tuple
        The tupled outputs.

    Returns
    -------
    result : tvm.relay.Expr
        The shape function expression.
    """
    return _make.shape_func(func, inputs, outputs, dependent)

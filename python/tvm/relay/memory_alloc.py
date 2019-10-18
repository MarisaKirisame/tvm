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
from .expr_functor import ExprMutator
from .scope_builder import ScopeBuilder
from . import transform
from .module import Module
from . import op, ty, expr
from .. import nd, TVMType, register_func
from ..target import current_target
from .. import convert
from .backend import compile_engine
import numpy as np

def is_primitive(call):
    return hasattr(call.op, 'attrs') and int(call.op.attrs.Primitive) == 1

class LinearizeRetType:
    """A linear view of a Relay type, handles a linear order
       for nested tuples, and tensor types.
    """
    def __init__(self, ty):
        self.ty = ty

    def unpack(self):
        def _unpack(typ, out):
            # TODO(@jroesch): replace with new flattening pass
            if isinstance(typ, ty.TensorType):
                out.append(typ)
            elif isinstance(typ, ty.TupleType):
                for field_ty in typ.fields:
                    _unpack(field_ty, out)
            else:
                raise Exception(f"unsupported Relay type: {typ}")

        output = []
        _unpack(self.ty, output)
        return output

    def pack(self, seq):
        def _pack(value, typ, out):
            if isinstance(typ, ty.TensorType):
                out.append(value)
            elif isinstance(typ, ty.TupleType):
                tuple_out = []
                for i, field_ty in enumerate(typ.fields):
                    _pack(value[i], field_ty, tuple_out)
                out.append(expr.Tuple(tuple_out))
            else:
                raise Exception(f"unsupported Relay type: {typ}")

        if len(seq) == 1:
            return seq[0]
        else:
            out = []
            _pack(seq, self.ty, out)
            assert len(out) == 1, "must return fully packed type"
            return out[0]

def infer_type(expr):
    m = Module.from_expr(expr)
    m = transform.InferType()(m)
    return m['main']

class ExplictAlloc(ExprMutator):
    def __init__(self, target_host):
        self.invoke_tvm = op.get('memory.invoke_tvm_op')
        self.alloc_storage = op.memory.alloc_storage
        self.alloc_tensor = op.memory.alloc_tensor
        self.shape_func = op.memory.shape_func
        self.scopes = [ScopeBuilder()]
        self.target_host = target_host
        self.compute_dtype = "int64"
        super().__init__()

    def current_scope(self):
        return self.scopes[-1]

    def shape_of(self, e):
        return op.shape_of(e, self.compute_dtype)
        # inp = expr.var("t1", type_annotation=e.checked_type)
        # shape_of = expr.Function([inp], op.shape_of(inp, self.compute_dtype))
        # one = convert(1)
        # # This is kind of a hack, need to think about this.
        # shape_of = shape_of.set_attribute("Primitive", one)
        # call = expr.Call(shape_of, [e])
        # func = infer_type(expr.Function([e], call))
        # ctype = func.body.checked_type
        # result = expr.bind(func.body, { func.params[0]: e })
        # result._checked_type_ = ctype
        # return result

    # def compute_storage(self, tensor_type):
    #     dtype = TVMType(tensor_type.dtype)
    #     shape = [int(sh) for sh in tensor_type.shape]
    #     if TVMType.CODE2STR[dtype.type_code] == 'float':
    #         dtype_size = dtype.bits * dtype.lanes
    #         size = 1
    #         for dim in shape:
    #             size *= dim
    #         size *= dtype_size
    #         return expr.const(size, dtype='int32')
    #     else:
    #         raise Exception("...")

    def visit_tuple(self, tuple):
        scope = self.current_scope()
        new_fields = []
        for field in tuple.fields:
            field = self.visit(field)
            if isinstance(field, expr.Constant):
                field = scope.let('const', field)
            new_fields.append(field)
        return expr.Tuple(new_fields)

    def compute_alignment(self, dtype):
        dtype = TVMType(dtype)
        align = (dtype.bits // 8) * dtype.lanes
        # MAGIC CONSTANT FROM device_api.h
        if align < 64:
            align = 64

        return expr.const(align, dtype="int64")

    def compute_storage_in_relay(self, shape, dtype):
        dtype = TVMType(dtype)
        els = op.prod(shape)
        num = expr.const(dtype.bits * dtype.lanes, self.compute_dtype) + expr.const(7, self.compute_dtype)
        div = expr.const(8, self.compute_dtype)
        return els * (num / div)

    def compute_storage(self, tensor_type):
        dtype = TVMType(tensor_type.dtype)
        shape = [int(sh) for sh in tensor_type.shape]
        size = 1
        for i in range(len(shape)):
            size *= shape[i]
        size *= (dtype.bits * dtype.lanes + 7) // 8
        return expr.const(size, dtype=self.compute_dtype)

    def make_static_allocation(self, scope, tensor_type, i):
        shape = [int(sh) for sh in tensor_type.shape]
        if not len(shape):
            shape = expr.const(np.array([]).astype(self.compute_dtype), dtype=self.compute_dtype)
        else:
            shape = expr.const(np.array(shape), dtype=self.compute_dtype)
        size = self.compute_storage(tensor_type)
        alignment = self.compute_alignment(tensor_type.dtype)
        dtype = tensor_type.dtype
        sto = scope.let(f"storage_{i}", self.alloc_storage(size, alignment, dtype))
        tensor = self.alloc_tensor(sto, shape, dtype)
        return scope.let(f"tensor_{i}", tensor)

    def visit_let(self, let):
        scope = ScopeBuilder()

        self.scopes.append(scope)
        while isinstance(let, expr.Let):
            new_val = self.visit(let.value)
            scope.let(let.var, new_val)
            let = let.body

        new_body = self.visit(let)
        scope.ret(new_body)
        self.scopes.pop()

        return scope.get()

    def visit_call(self, call):
        if is_primitive(call):
            # Because we are in ANF we do not need to visit the arguments.
            scope = self.current_scope()
            new_args = [self.visit(arg) for arg in call.args]
            ins = expr.Tuple(new_args)
            ret_type = call.checked_type

            if ret_type.is_dynamic():
                assert isinstance(ret_type, ty.TensorType)
                shape_func_ins = []
                engine = compile_engine.get()
                cfunc = engine.lower_shape_func(call.op, self.target_host)
                param_type = int(cfunc.shape_func_param_states[0])
                for state in cfunc.shape_func_param_states:
                    assert param_type == int(state)

                for i, arg in enumerate(new_args):
                    # Pass Shapes
                    if param_type == 2:
                        sh_of = self.visit(self.shape_of(arg))
                        shape_func_ins.append(scope.let(f"in_shape_{i}", sh_of))
                    # Pass Inputs
                    elif param_type == 1:
                        new_arg = self.visit(arg)
                        shape_func_ins.append(scope.let(f"in_shape_{i}", new_arg))
                    else:
                        raise Exception("error")

                if param_type == 2:
                    dependent = False
                elif param_type == 1:
                    dependent = True

                out_shapes = []
                for i, out in enumerate(cfunc.outputs):
                    tt = ty.TensorType(out.shape, out.dtype)
                    alloc = self.make_static_allocation(scope, tt, "TODO")
                    # alloc = self.visit(alloc.astype('int32'))
                    # alloc = scope.let("cast", alloc)
                    alloc = scope.let(f"shape_func_out_{i}", alloc)
                    out_shapes.append(alloc)

                shape_call = self.shape_func(call.op, expr.Tuple(shape_func_ins), expr.Tuple(out_shapes), dependent=dependent)
                scope.let("shape_func", shape_call)

                out_types = []
                out_types.append(call.checked_type)

                storages = []
                for out_shape, out_type in zip(out_shapes, out_types):
                    size = self.compute_storage_in_relay(out_shape, out_type.dtype)
                    alignment = self.compute_alignment(out_type.dtype)
                    sto = scope.let(f"storage_{i}", self.alloc_storage(size, alignment, out_type.dtype))
                    storages.append(sto)

                outs = []
                for i, (out_shape, out_type, storage) in enumerate(zip(out_shapes, out_types, storages)):
                    alloc = self.alloc_tensor(storage, out_shape, out_type.dtype, out_type.shape)
                    alloc = scope.let(f"out_{i}", alloc)
                    outs.append(alloc)

                invoke = self.invoke_tvm(call.op, ins, expr.Tuple(outs))
                scope.let("", invoke)
                return outs[0]
            else:
                view = LinearizeRetType(ret_type)
                out_tys = view.unpack()

                outs = []
                for i, out_ty in enumerate(out_tys):
                    out = self.make_static_allocation(scope, out_ty, i)
                    outs.append(out)

                output = expr.Tuple(outs)
                invoke = self.invoke_tvm(call.op, ins, output)
                scope.let("", invoke)
                return view.pack(output)
        else:
            return super().visit_call(call)

@transform.function_pass(opt_level=0)
class ManifestAlloc:
    def __init__(self, target_host):
        self.target_host = target_host

    def transform_function(self, func, mod, ctx):
         # Is there a way to do one shot initilization?
        mod.import_from_std("core.rly")
        ea = ExplictAlloc(self.target_host)
        r = ea.visit(func)
        print("After", r)
        return r

register_func("relay.transform.ManifestAlloc", ManifestAlloc)

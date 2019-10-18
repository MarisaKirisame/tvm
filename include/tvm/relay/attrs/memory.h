/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/attrs/memory.h
 * \brief Attributes for memory operators.
 */
#ifndef TVM_RELAY_ATTRS_MEMORY_H_
#define TVM_RELAY_ATTRS_MEMORY_H_

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the device annotation operators.
 */
struct AllocTensorAttrs : public tvm::AttrsNode<AllocTensorAttrs> {
  tvm::relay::Constant const_shape;
  Array<IndexExpr> assert_shape;
  DataType dtype;

  TVM_DECLARE_ATTRS(AllocTensorAttrs, "relay.attrs.AllocTensorAttrs") {
    TVM_ATTR_FIELD(dtype)
      .describe(
         "The dtype of the tensor to allocate.")
      .set_default(Float(32, 1));
    TVM_ATTR_FIELD(const_shape)
      .describe(
         "The shape if constant used to aid in type inference.");
    TVM_ATTR_FIELD(assert_shape)
      .describe(
         "The shape to cast the return type of the allocation to, used to specify the shape obtained via further analysis.");
  }
};

/*!
 * \brief Options for the device annotation operators.
 */
struct ShapeFuncAttrs : public tvm::AttrsNode<ShapeFuncAttrs> {
  bool dependent{false};

  TVM_DECLARE_ATTRS(ShapeFuncAttrs, "relay.attrs.ShapeFuncAttrs") {
    TVM_ATTR_FIELD(dependent)
      .describe(
         "Wheather the shape function is input dependent.")
      .set_default(false);
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_MEMORY_H_

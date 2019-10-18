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
 * \file tvm/relay/attrs/annotation.h
 * \brief Attribute for annotation operators.
 */
#ifndef TVM_RELAY_ATTRS_ANNOTATION_H_
#define TVM_RELAY_ATTRS_ANNOTATION_H_

#include <tvm/attrs.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the device annotation operators.
 */
struct OnDeviceAttrs : public tvm::AttrsNode<OnDeviceAttrs> {
  int device_type;

  TVM_DECLARE_ATTRS(OnDeviceAttrs, "relay.attrs.OnDeviceAttrs") {
    TVM_ATTR_FIELD(device_type)
      .describe(
         "The virutal device/context type that an expression is annotated with.")
      .set_default(0);
  }
};

/*!
 * \brief Annotate an expression to be cast into specific data type.
 */
struct CastHintAttrs : public tvm::AttrsNode<CastHintAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastHintAttrs, "relay.attrs.CastHintAttrs") {
    TVM_ATTR_FIELD(dtype)
      .describe(
         "The data type denoted to be cast.");
  }
};

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
         "The virutal device/context type that an expression is annotated with.")
      .set_default(Float(32, 1));
    TVM_ATTR_FIELD(const_shape)
      .describe(
         "The virutal device/context type that an expression is annotated with.");
    TVM_ATTR_FIELD(assert_shape)
      .describe(
         "The virutal device/context type that an expression is annotated with.");
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
#endif  // TVM_RELAY_ATTRS_ANNOTATION_H_

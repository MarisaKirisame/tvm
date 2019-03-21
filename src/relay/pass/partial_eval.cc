/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file partial_eval.cc
 *
 * \brief Perform known computation in compile time.
 *
 * The partial evaluator try to do computation at compile time,
 * so it can generate code that do less work.
 * Additionally, it might open more chance for further optimization,
 * since the high level, structural part of the code (closure, reference, control flow)
 * might get partially evaluated away, and the subsequent optimization (for example, kernel fusion)
 * can reason across those structural code as it got removed.
 * In the extreme case, partial evaluation can even turn the whole program
 * into pure first order computation with no control.
 * In such a case, we can compile the whole computation onto SIMD Instruction/GPU/FPGA,
 * and get insane speedup.
 *
 * It works by making the following modifications to the standard relay interpreter:
 *
 * 0: The values become partially static value.
 * Since we cannot know the value of every term at compile time,
 * Term might get partially evaluated to 'Unknown Value'.
 * Every partially static value is, hence,
 * a static fragment that might not be there (partially static),
 * and a dynamic fragment that is semantically equivalent to the original term,
 * so the unknown part will be computed at runtime, using the dynamic fragment.
 *
 * 1: The interpreter get paired with a LetList, which preserves ANF for the generated code.
 * More specifically, we require that all dynamic is an atom.
 * This avoid code duplication (which is both inefficient and incorrect) as atom has constant size,
 * and allow us to not handle capture-avoidance substitution (as atom has no binder).
 *
 * 2: The store get reified.
 * Instead of Reference having mutable field, Reference only has an unique identifier.
 * There will be a mutable mapping of id to partially static value, called the store.
 * This allow us to rollback the store:
 * when a path may or may not get executed, we copy the store, recurse with a copy,
 * and when we are done, we continue with the old store, so those effect 'do not happened'.
 * We do this in if else, pattern matching, and in function,
 * as, when we see a function, we partially evaluate it with all the argument as dynamic,
 * to generate efficient dynamic for that function.
 *
 * 3: The code generated will reuse binding (although they are not shadowed),
 * so we has to dedup them.
 *
 * 4: It will also generate lots of dead code, so we will feed it through the dead code eliminator.
 *
 * There are some rooms for obvious improvement.
 *
 * 0: The partial evaluator treat global variable as opaque.
 * Doing PartialEval on a module level will solve this.
 *
 * 1: The partial evaluator assume all functions as terminating.
 * We need to has a max_expand parameter that shrink on every compile time evaluation,
 * to make sure PE does not infinite loop.
 * Additionally, we might add a termination analysis pass that lift this requirement
 * for function that analysis found terminating.
 * A binding time analysis might also be nice.
 *
 * 2: Every time an unknown effect happened, we clear the whole store.
 * It is too conservative: if a local reference is created (and do not get passed outside),
 * An unknown global function call/global reference write can not modify it.
 * We can pair PE with escape analysis/alias analysis.
 *
 * 3: We assume all unknown code has effect. Doing effect analysis can make the store more precise.
 *
 * 4: When doing pattern matching, we can simplify the match even for dynamic case.
 * Right now it is all or nothing: either a complete match, or the original dynamic code.
 * Instead, we can get a match tree, pair it with the data and evaluate it to a normal form.
 * We then can reify the result.
 *
 * Luckily, this three issues do not effect the correctness of the algorithm.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/interpreter.h>
#include "pass_util.h"
#include "let_list.h"

namespace tvm {
namespace relay {

using namespace runtime;

struct StaticNode {
  virtual ~StaticNode() { }
  template <typename T>
  T& get() {
    auto ret = dynamic_cast<T*>(this);
    CHECK(ret) << "cannot downcast";
    return *ret;
  }
  template <typename T>
  T* try_get() {
    auto ret = dynamic_cast<T*>(this);
    return ret;
  }
};

using Static = std::shared_ptr<StaticNode>;

struct PStaticNode;

using PStatic = std::shared_ptr<PStaticNode>;

struct STupleNode : StaticNode {
  std::vector<PStatic> fields;
  explicit STupleNode(const std::vector<PStatic>& fields) : fields(fields) { }
};

Static STuple(const std::vector<PStatic>& fields) {
  return std::make_shared<STupleNode>(fields);
}

struct STensorNode : StaticNode {
  runtime::NDArray data;
  explicit STensorNode(const NDArray& data) : data(data) { }
};

Static STensor(const NDArray& data) {
  return std::make_shared<STensorNode>(data);
}

struct SConstructorNode : StaticNode {
  Constructor constructor;
  std::vector<PStatic> fields;
  SConstructorNode(const Constructor& constructor, const std::vector<PStatic>& fields) :
    constructor(constructor), fields(fields) { }
};

Static SConstructor(const Constructor& constructor, const std::vector<PStatic>& fields) {
  return std::make_shared<SConstructorNode>(constructor, fields);
}

struct SRefNode : StaticNode { };  // we will use the pointer as the guid for hashing

Static SRef() {
  return std::make_shared<SRefNode>();
}

using Clos = std::function<PStatic(const std::vector<PStatic>&,
                                   const Attrs&,
                                   const Array<Type>&,
                                   LetList*)>;

struct SClosNode : StaticNode {
  Clos func;
  explicit SClosNode(const Clos& func) : func(func) { }
};

Static SClos(const Clos& func) {
  return std::make_shared<SClosNode>(func);
}

struct PStaticNode {
  Static pstatic;  // may be null
  Expr dynamic;
  PStaticNode(const Static& pstatic, const Expr& dynamic) : pstatic(pstatic), dynamic(dynamic) { }
  explicit PStaticNode(const Expr& dynamic) : PStaticNode(Static(), dynamic) { }
};

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::Value.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  std::unordered_map<Var, PStatic, NodeHash, NodeEqual> locals;
  explicit Frame(const std::unordered_map<Var, PStatic, NodeHash, NodeEqual>& locals) :
    locals(locals) { }
  Frame() = default;
};

struct Environment {
  std::list<Frame> env;
  Environment() : env({Frame()}) { }

  struct RAII {
    Environment* env;
    explicit RAII(Environment* env) : env(env) {
      env->env.push_back(Frame());
    }
    ~RAII() {
      env->env.pop_back();
    }
  };

  template<typename T>
  T Extend(const std::function<T()>& cont) {
    RAII raii(this);
    return cont();
  }

  void Insert(const Var& v, const PStatic& ps) {
    env.back().locals[v] = ps;
  }

  // return null if not found
  PStatic Lookup(const Var& v) {
    auto rit = env.rbegin();
    while (rit != env.rend()) {
      if (rit->locals.find(v) != rit->locals.end()) {
        return rit->locals.find(v)->second;
      }
      ++rit;
    }
    return PStatic();
  }
};

/*!
 * \brief As our store require rollback, we implement it as a frame.
 * every time we need to copy the store, a new frame is insert.
 * every time we roll back, a frame is popped.
 */
struct StoreFrame {
  std::unordered_map<SRefNode*, PStatic> store;
  /*! \brief on unknown effect, history_valid is set to true to signal above frame is outdated */
  bool history_valid = true;
  explicit StoreFrame(const std::unordered_map<SRefNode*, PStatic>& store) : store(store) { }
  StoreFrame() = default;
};

struct Store {
  std::list<StoreFrame> store;
  Store() : store({StoreFrame()}) { }

  struct RAII {
    Store* store;
    explicit RAII(Store* store) : store(store) {
      store->store.push_back(StoreFrame());
    }
    ~RAII() {
      store->store.pop_back();
    }
  };

  template<typename T>
  T Extend(const std::function<T()>& cont) {
    RAII raii(this);
    return cont();
  }

  void Insert(SRefNode* r, const PStatic& ps) {
    store.back().store[r] = ps;
  }

  void Invalidate() {
    store.back().history_valid = false;
  }

  // return null if not found
  PStatic Lookup(SRefNode* r) {
    auto rit = store.rbegin();
    while (rit != store.rend()) {
      if (rit->store.find(r) != rit->store.end()) {
        return rit->store.find(r)->second;
      }
      if (rit->history_valid) {
        ++rit;
      } else {
        return PStatic();
      }
    }
    return PStatic();
  }
};

PStatic HasStatic(const Static& stat, const Expr& dynamic) {
  return std::make_shared<PStaticNode>(stat, dynamic);
}

PStatic NoStatic(const Expr& dynamic) {
  return std::make_shared<PStaticNode>(dynamic);
}

enum struct MatchStatus {
  Match, NoMatch, Unknown
};

bool StatefulOp(const Expr& e) {
  static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");
  struct StatefulOpVisitor : ExprVisitor {
    bool stateful = false;
    void VisitExpr_(const OpNode* op) {
      stateful = stateful || op_stateful.get(GetRef<Op>(op), false);
    }
  };
  StatefulOpVisitor sov;
  sov(e);
  return sov.stateful;
}

using FInterpreter = runtime::TypedPackedFunc<Value(Expr)>;

FInterpreter CPUInterpreter() {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  Target target = Target::create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  BuildConfigContext fresh_build_ctx(build_config());

  return CreateInterpreter(Module(nullptr), ctx, target);
}

struct PartialEvaluator : ExprFunctor<PStatic(const Expr& e, LetList* ll)>,
                          PatternFunctor<MatchStatus(const Pattern&, const PStatic&)> {
  Environment env;
  Store store;
  DLContext context;
  FInterpreter executor = CPUInterpreter();
  PStatic VisitExpr_(const ConstantNode* op, LetList* ll) final {
    return HasStatic(STensor(op->data.CopyTo(context)), ll->Push(GetRef<Expr>(op)));
  }

  PStatic VisitExpr_(const TupleNode* op, LetList* ll) final {
    std::vector<PStatic> value;
    tvm::Array<Expr> expr;
    for (const Expr& e : op->fields) {
      PStatic ps = VisitExpr(e, ll);
      value.push_back(ps);
      expr.push_back(ps->dynamic);
    }
    return HasStatic(STuple(value), ll->Push(TupleNode::make(expr)));
  }

  PStatic VisitExpr_(const TupleGetItemNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->tuple, ll);
    if (ps->pstatic) {
      return ps->pstatic->get<STupleNode>().fields[op->index];
    } else {
      return NoStatic(ll->Push(TupleGetItemNode::make(ps->dynamic, op->index)));
    }
  }

  PStatic VisitExpr_(const VarNode* op, LetList* ll) final {
    PStatic ret = env.Lookup(GetRef<Var>(op));
    CHECK(ret);
    return ret;
  }

  PStatic VisitExpr_(const GlobalVarNode* op, LetList* ll) final {
    return NoStatic(GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const LetNode* op, LetList* ll) final {
    env.Insert(op->var, VisitExpr(op->value, ll));
    return VisitExpr(op->body, ll);
  }

  PStatic VisitExpr_(const IfNode* op, LetList* ll) final {
    PStatic c = VisitExpr(op->cond, ll);
    if (c->pstatic) {
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = c->pstatic->get<STensorNode>().data.CopyTo(cpu_ctx);
      CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
      // TODO(@jroesch, @MK): Refactor code into helper from DCE.
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return VisitExpr(op->true_branch, ll);
      } else {
        return VisitExpr(op->false_branch, ll);
      }
    } else {
      Expr t = store.Extend<Expr>([&](){ return VisitExpr(op->true_branch, ll)->dynamic; });
      Expr f = store.Extend<Expr>([&](){ return VisitExpr(op->false_branch, ll)->dynamic; });
      store.Invalidate();
      return NoStatic(ll->Push(IfNode::make(c->dynamic, t, f)));
    }
  }

  PStatic VisitExpr_(const RefCreateNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->value, ll);
    Static r = SRef();
    store.Insert(&r->get<SRefNode>(), ps);
    return HasStatic(r, RefCreateNode::make(ps->dynamic));
  }

  PStatic VisitExpr_(const RefWriteNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    PStatic v = VisitExpr(op->value, ll);
    if (r->pstatic) {
      store.Insert(&r->pstatic->get<SRefNode>(), v);
    } else {
      store.Invalidate();
    }
    return HasStatic(STuple({}), RefWriteNode::make(r->dynamic, v->dynamic));
  }

  PStatic VisitExpr_(const RefReadNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    if (r->pstatic) {
      PStatic ret = store.Lookup(&r->pstatic->get<SRefNode>());
      if (ret) {
        return ret;
      }
    }
    return NoStatic(RefReadNode::make(r->dynamic));
  }

  PStatic VisitExpr_(const CallNode* op, LetList* ll) final {
    PStatic f = VisitExpr(op->op, ll);
    std::vector<PStatic> x;
    tvm::Array<Expr> x_dyn;
    for (const Expr& e : op->args) {
      PStatic ps = VisitExpr(e, ll);
      x.push_back(ps);
      x_dyn.push_back(ps->dynamic);
    }
    if (f->pstatic) {
      return f->pstatic->get<SClosNode>().func(x, op->attrs, op->type_args, ll);
    } else {
      store.Invalidate();
      return NoStatic(CallNode::make(f->dynamic, x_dyn, op->attrs, op->type_args));
    }
  }

  PStatic VisitExpr_(const FunctionNode* op, LetList* ll) final {
    // TODO(@M.K.): TypeSubst is removed. Implement it.
    if (op->IsPrimitive()) {
      return HasStatic(SClos(ConstEvaluateClos(GetRef<Expr>(op), ll)), GetRef<Expr>(op));
    } else {
      Clos f = [=](const std::vector<PStatic>& pv,
                   const Attrs& attrs,
                   const tvm::Array<Type>& type_args,
                   LetList* ll) {
        return env.Extend<PStatic>([&]() {
            CHECK_EQ(pv.size(), op->params.size());
            for (size_t i = 0; i < pv.size(); ++i) {
              env.Insert(op->params[i], pv[i]);
            }
            tvm::Map<TypeVar, Type> subst;
            for (size_t i = 0; i < type_args.size(); ++i) {
              subst.Set(op->type_params[i], type_args[i]);
            }
            for (size_t i = type_args.size(); i < op->type_params.size(); ++i) {
              subst.Set(op->type_params[i], Type());
            }
            return VisitExpr(TypeSubst(op->body, subst), ll);
          });
      };
      Expr dyn = store.Extend<Expr>([&]() {
          return FunctionNode::make(op->params, LetList::With([&](LetList* ll) {
                std::vector<PStatic> pv;
                for (const auto& v : op->params) {
                  pv.push_back(NoStatic(v));
                }
                tvm::Array<Type> type_args;
                for (const auto& tp : op->type_params) {
                  type_args.push_back(tp);
                }
                return f(pv, Attrs(), type_args, ll)->dynamic;
              }), op->ret_type, op->type_params, op->attrs);
        });
      return HasStatic(SClos(f), dyn);
    }
  }

  Expr Reflect(const PStatic& st) {
    if (const STensorNode* op = st->pstatic->try_get<STensorNode>()) {
      return ConstantNode::make(op->data);
    } else if (const STupleNode* op = st->pstatic->try_get<STupleNode>()) {
      tvm::Array<Expr> fields;
      for (const PStatic& field : op->fields) {
        fields.push_back(Reflect(field));
      }
      return TupleNode::make(fields);
    } else {
      LOG(ERROR) << "Unknown case";
      throw;
    }
  }

  PStatic Reify(const Value& v, LetList* ll) const {
    if (const TensorValueNode* op = v.as<TensorValueNode>()) {
      return HasStatic(STensor(op->data), ll->Push(ConstantNode::make(op->data)));
    } else if (const TupleValueNode* op = v.as<TupleValueNode>()) {
      std::vector<PStatic> fields;
      tvm::Array<Expr> fields_dyn;
      for (const Value& field : op->fields) {
        PStatic ps = Reify(field, ll);
        fields.push_back(ps);
        fields_dyn.push_back(ps->dynamic);
      }
      return HasStatic(STuple(fields), ll->Push(TupleNode::make(fields_dyn)));
    } else {
      LOG(ERROR) << "Unknown case";
      throw;
    }
  }

  // Constant evaluate a expression.
  PStatic ConstEvaluate(const Expr& expr, LetList* ll) {
    Expr infered = InferType(expr, Module(nullptr));
    Expr fused = FuseOps(expr, 0);
    Expr fused_infered = InferType(expr, Module(nullptr));
    return Reify(executor(fused_infered), ll);
  }

  Clos ConstEvaluateClos(const Expr& expr, LetList* ll) {
    return [=](const std::vector<PStatic>& pv,
                 const Attrs& attrs,
                 const tvm::Array<Type>& type_args,
                 LetList* ll) {
      tvm::Array<Expr> ns_args;
      for (const PStatic& ps : pv) {
        ns_args.push_back(ps->dynamic);
      }
      PStatic ns = NoStatic(CallNode::make(expr, ns_args, attrs, type_args));
      if (!StatefulOp(expr)) {
        tvm::Array<Expr> args;
        for (const PStatic& ps : pv) {
          if (ps->pstatic) {
            args.push_back(Reflect(ps));
          } else {
            return ns;
          }
        }
        return ConstEvaluate(CallNode::make(expr, args, attrs, type_args), ll);
      }
      return ns;
    };
  }

  PStatic VisitExpr_(const OpNode* op, LetList* ll) final {
    return HasStatic(SClos(ConstEvaluateClos(GetRef<Expr>(op), ll)), GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const ConstructorNode* op, LetList* ll) final {
    Clos f = [=](const std::vector<PStatic>& pv,
                 const Attrs& attrs,
                 const tvm::Array<Type>& type_args,
                 LetList* ll) {
      Constructor c = GetRef<Constructor>(op);
      tvm::Array<Expr> dyn;
      for (const PStatic& ps : pv) {
        dyn.push_back(ps->dynamic);
      }
      return HasStatic(SConstructor(c, pv), ll->Push(CallNode::make(c, dyn)));
    };
    return HasStatic(SClos(f), GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const MatchNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->data, ll);
    return env.Extend<PStatic>([&]() {
        for (const Clause& c : op->clauses) {
          switch (VisitPattern(c->lhs, ps)) {
          case MatchStatus::Match:
            return VisitExpr(c->rhs, ll);
          case MatchStatus::NoMatch:
            continue;
          case MatchStatus::Unknown:
            tvm::Array<Clause> clauses;
            for (const Clause& c : op->clauses) {
              clauses.push_back(ClauseNode::make(c->lhs,
                                                 store.Extend<Expr>([&]() {
                                                     return VisitExpr(c->rhs, ll)->dynamic;
                                                   })));
            }
            store.Invalidate();
            return NoStatic(MatchNode::make(ps->dynamic, clauses));
          }
        }
        LOG(ERROR) << "No case Match";
        throw;
      });
  }

  MatchStatus VisitPattern_(const PatternWildcardNode* op, const PStatic& ps) final {
    return MatchStatus::Match;
  }

  MatchStatus VisitPattern_(const PatternVarNode* op, const PStatic& ps) final {
    env.Insert(op->var, ps);
    return MatchStatus::Match;
  }

  MatchStatus VisitPattern_(const PatternConstructorNode* op, const PStatic& ps) final {
    if (ps->pstatic) {
      const SConstructorNode& scn = ps->pstatic->get<SConstructorNode>();
      CHECK_NE(op->constructor->tag, -1);
      CHECK_NE(scn.constructor->tag, -1);
      if (op->constructor->tag == scn.constructor->tag) {
        // todo(M.K.): should use ptr equality but it is broken
        CHECK_EQ(op->patterns.size(), scn.fields.size());
        MatchStatus current_match_status = MatchStatus::Match;
        for (size_t i = 0; i < op->patterns.size(); ++i) {
          MatchStatus ms = VisitPattern(op->patterns[i], scn.fields[i]);
          switch (ms) {
          case MatchStatus::Match:
            continue;
          case MatchStatus::NoMatch:
            current_match_status = MatchStatus::NoMatch;
            continue;
          case MatchStatus::Unknown:
            if (current_match_status == MatchStatus::Match) {
              current_match_status = MatchStatus::Unknown;
            }
          }
        }
        return current_match_status;
      }
      return MatchStatus::NoMatch;
    } else {
      return MatchStatus::Unknown;
    }
  }
};

Expr DeDup(const Expr& e) {
  return e;
}

Expr PartialEval(const Expr& e) {
  PartialEvaluator pe;
  Expr res = LetList::With([&](LetList* ll) {
      return DeDup(pe.VisitExpr(e, ll)->dynamic);
    });
  if (const FunctionNode* op = e.as<FunctionNode>()) {
    tvm::Array<Var> params;
    tvm::Array<Expr> params_expr;
    for (const Var& v : op->params) {
      Var fresh_v = VarNode::make(v->name_hint(), v->type_annotation);
      params.push_back(fresh_v);
      params_expr.push_back(fresh_v);
    }
    tvm::Array<TypeVar> type_params;
    tvm::Array<Type> type_params_type;
    for (const TypeVar& tv : op->type_params) {
      TypeVar fresh_tv = TypeVarNode::make(tv->var->name_hint, tv->kind);
      type_params.push_back(fresh_tv);
      type_params_type.push_back(fresh_tv);
    }
    return FunctionNode::make(params,
                              CallNode::make(res, params_expr, Attrs(), type_params_type),
                              Type(),
                              type_params,
                              op->attrs);
  } else {
    return res;
  }
}

TVM_REGISTER_API("relay._ir_pass.partial_eval")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = PartialEval(args[0]);
  });

}  // namespace relay
}  // namespace tvm

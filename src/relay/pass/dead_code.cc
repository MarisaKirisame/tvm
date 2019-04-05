/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file dead_code.cc
 *
 * \brief Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 *
 * Also, Dead Code Eliminator has to take into account of effect -
 * Call to foreign function should not be eliminated.
 * Write to reference should not be eliminated if that reference is used.
 *
 * To do this we implement a simple escape analysis.
 * We abstract Reference Value point to StoreId.
 * Each RefCreate get a unique StoreId,
 * And also assign each parameter a unique StoreId (as they might has/contain Ref).
 * We then create a map of Expr -> Set StoreId, which record what StoreId Expr might depend on.
 * The map is ran until a Fixpoint (it will terminate as there are finite StoreId.).
 * The StoreId inside the inputs and the body are all the StoreId that is alive,
 * and effect to other StoreId can be removed.
 *
 * We choose to implement StoreId as Expr for simplicity.
 *
 * Whenever a function is called, or a reference is written into,
 * We make the set of reference inside depend on that call/write.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include "let_list.h"

namespace tvm {
namespace relay {

using Sid = Expr;
using SidSet = std::unordered_set<Sid, NodeHash, NodeEqual>;
using ExprSet = std::unordered_set<Expr, NodeHash, NodeEqual>;
template<typename T>
using ExprMap = std::unordered_map<Expr, T, NodeHash, NodeEqual>;
template<typename X>
using VarMap = std::unordered_map<Var, X, VarHash, VarEqual>;
using VarSet = std::unordered_set<Var, VarHash, VarEqual>;

struct EscapeAnalysis : ExprFunctor<void(const Expr&, const Expr&)>,
                        PatternFunctor<void(const Pattern&, const Expr&)> {
  ExprMap<SidSet> map_;
  SidSet live_sid_;
  ExprSet root_expr_;
  bool converge = false;
  bool HasEffect(const Expr& e) {
    struct EffectVisitor : ExprVisitor {
      EscapeAnalysis* ea_;
      explicit EffectVisitor(EscapeAnalysis* ea) : ea_(ea) { }
      bool has_effect = false;
      void Touch(const Expr& e) {
        for (const Sid& s: ea_->Get(e)) {
          has_effect |= (ea_->live_sid_.count(s) > 0);
          if (ea_->live_sid_.count(s) > 0) {
            std::cout << "HAS EFFECT" << std::endl;
            return;
          }
        }
      }
      void VisitExpr_(const RefReadNode* op) final {
        Touch(op->ref);
        VisitExpr(op->ref);
      }
      void VisitExpr_(const RefWriteNode* op) final {
        Touch(op->ref);
        VisitExpr(op->ref);
        VisitExpr(op->value);
      }
      void VisitExpr_(const CallNode* op) final {
        // The args contain same sid as op, so no need to touch them.
        Touch(op->op);
        VisitExpr(op->op);
        for (const Expr& arg: op->args) {
          VisitExpr(arg);
        }
      }
      void VisitExpr_(const FunctionNode* op) final { }
    };
    std::cout << "CHECK EFFECT:" << e << std::endl;
    EffectVisitor ev(this);
    ev(e);
    return ev.has_effect;
  }
  explicit EscapeAnalysis(const Expr& e) {
    for (const Var& v: FreeVars(e)) {
      AllocRoot(v);
    }
    while (!converge) {
      converge = true;
      Analysis(e);
    }
    Alive(e);
    for (const Expr& r: root_expr_) {
      Alive(r);
    }
    while (!converge) {
      converge = true;
      std::vector<Sid> live_sid_old;
      for (const Sid& s: live_sid_) {
        live_sid_old.push_back(s);
      }
      for (const Sid& s: live_sid_old) {
        Alive(s);
      }
    }
  }
  void Alive(const Expr& e) {
    for (const Sid& s: Get(e)) {
      if (live_sid_.count(s) == 0) {
        converge = false;
        live_sid_.insert(s);
      }
    }
  }
  void Analysis(const Expr& e) {
    VisitExpr(e, e);
  }
  ExprSet& Get(const Expr& e) {
    if (map_.count(e) == 0) {
      map_.insert({e, ExprSet()});
    }
    return map_.at(e);
  }
  std::vector<Sid> Range(const Expr& e) {
    std::vector<Sid> ret;
    for (const auto& x: Get(e)) {
      ret.push_back(x);
    }
    return ret;
  }
  void Insert(const Expr& from, const Expr& to) {
    ExprSet& x = Get(from);
    if (x.count(to) == 0) {
      converge = false;
      x.insert(to);
    }
  }
  void Join(const Expr& from, const Expr& to) {
    for (const Expr& e: Range(to)) {
      Insert(from, e);
    }
  }
  void Write(const Expr& from, const Expr& to) {
    for (const Expr& e: Range(from)) {
      Join(e, to);
    }
  }
  void Alloc(const Expr& e) {
    Insert(e, e);
  }
  void Root(const Expr& e) {
    root_expr_.insert(e);
  }
  void AllocRoot(const Expr& e) {
    Alloc(e);
    Root(e);
  }
  void Depend(const Expr& val, const Expr& on) {
    Analysis(on);
    Join(val, on);
  }
  void VisitExpr_(const RefCreateNode* op, const Expr& e) final {
    AllocRoot(e);
    Depend(e, op->value);
  }
  void VisitExpr_(const RefWriteNode* op, const Expr& e) final {
    Write(e, op->ref);
    Analysis(op->ref);
    Analysis(op->value);
  }
  void VisitExpr_(const FunctionNode* op, const Expr& e) final {
    for (const Var& v: op->params) {
      AllocRoot(v);
    }
    Root(op->body);
    Depend(e, op->body);
  }
  void VisitExpr_(const CallNode* op, const Expr& e) final {
    std::vector<Expr> exprs;
    Depend(e, op->op);
    exprs.push_back(op->op);
    for (const Expr& arg: op->args) {
      Depend(e, arg);
      exprs.push_back(arg);
    }
    for (size_t i = 0; i < exprs.size(); ++i) {
      for (size_t j = i + 1; j < exprs.size(); ++j) {
        Write(exprs[i], exprs[j]);
        Write(exprs[j], exprs[i]);
      }
    }
  }
  void RecordVar(const Var& v) {
    Get(v);
  }
  void VisitExpr_(const LetNode* op, const Expr& e) final {
    RecordVar(op->var);
    Depend(op->var, op->value);
    Depend(e, op->body);
  }
  // From here on the uninteresting case: just declare Depend on children
  void VisitExpr_(const VarNode* op, const Expr& e) final {
    CHECK_GT(map_.count(GetRef<Expr>(op)), 0);
  }
  void VisitExpr_(const ConstructorNode* op, const Expr& e) final { }
  void VisitExpr_(const OpNode* op, const Expr& e) final {
    // TODO(@M.K.): handle stateful op
  }
  void VisitExpr_(const ConstantNode* op, const Expr& e) final { }
  void VisitExpr_(const GlobalVarNode* op, const Expr& e) final { }
  void VisitExpr_(const MatchNode* op, const Expr& e) final {
    Depend(e, op->data);
    for (const Clause& c: op->clauses) {
      VisitPattern(c->lhs, op->data);
      Depend(e, c->rhs);
    }
  }
  void VisitPattern_(const PatternWildcardNode* op, const Expr& e) final { }
  void VisitPattern_(const PatternVarNode* op, const Expr& e) final {
    Depend(op->var, e);
  }
  void VisitPattern_(const PatternConstructorNode* op, const Expr& e) final {
    for (const Pattern& pat: op->patterns) {
      VisitPattern(pat, e);
    }
  }
  void VisitExpr_(const RefReadNode* op, const Expr& e) final {
    Depend(e, op->ref);
  }
  void VisitExpr_(const TupleNode* op, const Expr& e) final {
    for (const Expr& c: op->fields) {
      Depend(e, c);
    }
  }
  void VisitExpr_(const TupleGetItemNode* op, const Expr& e) final {
    Depend(e, op->tuple);
  }
  void VisitExpr_(const IfNode* op, const Expr& e) final {
    Depend(e, op->cond);
    Depend(e, op->true_branch);
    Depend(e, op->false_branch);
  }
};


// calculate the dependency graph from expression
class CalcDep : private ExprVisitor {
 public:
  explicit CalcDep(const Expr& v) {
    VisitExpr(v);
    return;
    count_ = false;
    while (!dead_worklist_.empty()) {
      Var dead = *(dead_worklist_.begin());
      dead_worklist_.erase(dead);
      CHECK_EQ(use_map_[dead], 0);
      if (expr_map_.count(dead) > 0) {
        LetRec([&]() { VisitExpr(expr_map_[dead]); }, dead);
      }
    }
  }

  bool Used(const Var& v) {
    return use_map_[v] > 0;
  }

  bool HasLet(const Var& v) {
    return (use_map_[v] > 1 || (use_map_[v] != 0 && letrec_set_.count(v) != 0));
  }

  Expr Map(const Var& v) {
    return expr_map_.count(v) == 0 ? Expr(v) : expr_map_[v];
  }

 private:
  VarMap<Expr> expr_map_;
  VarMap<size_t> use_map_;
  VarSet letrec_set_;
  bool count_ = true;
  VarSet dead_worklist_;
  VarSet current_letrec_;

  void LetRec(const std::function<void()>& func, const Var& v) {
    current_letrec_.insert(v);
    func();
    current_letrec_.erase(v);
  }

  void VisitExpr_(const LetNode* l) final {
    if (count_) {
      CHECK_EQ(expr_map_.count(l->var), 0);
      CHECK_EQ(use_map_.count(l->var), 0);
      expr_map_[l->var] = l->value;
      use_map_[l->var] = 0;
      dead_worklist_.insert(l->var);
      LetRec([&]() { VisitExpr(l->value); }, l->var);
    }
    VisitExpr(l->body);
  }

  void VisitExpr(const Expr& e) final {
    ExprFunctor<void(const Expr&)>::VisitExpr(e);
  }

  void VisitExpr_(const VarNode* v) final {
    Var var = GetRef<Var>(v);
    if (expr_map_.count(var) == 0) {
      return;
    }
    if (current_letrec_.count(var) == 0) {
      if (count_) {
        use_map_[var] += 1;
        dead_worklist_.erase(var);
      } else {
        CHECK_GT(use_map_[var], 0) << var;
        use_map_[var] -= 1;
        if (use_map_[var] == 0) {
          dead_worklist_.insert(var);
        }
      }
    } else {
      letrec_set_.insert(var);
    }
  }
};

class Eliminator : private ExprMutator {
 public:
  static Expr Eliminate(const Expr& e) {
    Eliminator elm(e);
    return elm(e);
  }
 private:
  EscapeAnalysis ea_;
  CalcDep cd_;
  explicit Eliminator(const Expr& e) : ea_(e), cd_(e) { }
  friend CalcDep;

  Expr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    return v;
    std::cout << v << " map to " << cd_.Map(v) << std::endl;
    return (cd_.Used(v) || ea_.HasEffect(cd_.Map(v))) ? v : cd_.Map(v);
  }

  Expr VisitExpr_(const LetNode* op) final {
    Var v = op->var;
    CHECK_EQ(cd_.Map(v), op->value);
    if (cd_.Used(v) || ea_.HasEffect(op->value)) {
      return LetNode::make(v, VisitExpr(op->value), VisitExpr(op->body));
    } else {
      return VisitExpr(op->body);
    }
  }
  //Expr VisitExpr_(const IfNode* op) final {
  //  return IfNode::make(op->cond, Descend(op->true_branch), Descend(op->false_branch));
  //}
};

Expr DeadCodeElimination(const Expr& e) {
  return Eliminator::Eliminate(e);
}

TVM_REGISTER_API("relay._ir_pass.dead_code_elimination")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = DeadCodeElimination(args[0]);
  });

}  // namespace relay
}  // namespace tvm

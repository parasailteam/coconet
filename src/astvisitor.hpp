#ifndef __ASTVISITOR_HPP__
#define __ASTVISITOR_HPP__

#include <iostream>
#include <set>

#include "ast.hpp"
#include "utils.hpp"
#include <unordered_map>

namespace ACCCDSLImpl {
class CloneAstVisitor : public AstVisitor{
protected:
    std::unordered_map<AstNodeImpl*, std::shared_ptr<AstNodeImpl>> origToCloneMap_;
public:
    #define checkMap \
        if (origToCloneMap_.find(&node) != origToCloneMap_.end()) return;

    CloneAstVisitor() {}

    template<typename T>
    void addToMap(T& node, T* x) {
        origToCloneMap_[&node] = std::shared_ptr<T>(x);
    }

    const std::unordered_map<AstNodeImpl*, std::shared_ptr<AstNodeImpl>>& origToCloneMap() {return origToCloneMap_;}
    std::shared_ptr<AstNodeImpl> clone(AstNodeImpl& node)
    {
        if (origToCloneMap_.find(&node) == origToCloneMap_.end()){
            node.accept(*this);
        }
        return origToCloneMap_[&node];
    }
    std::shared_ptr<AstNodeImpl> clone(std::shared_ptr<AstNodeImpl> node)
    {
        return clone(*node.get());
    }
    virtual void visit(TensorImpl& node)
    {
        checkMap
        TensorImpl* t = new TensorImpl(node);
        addToMap(node, t);
    }
    virtual void visit(AllReduceImpl& node)
    {
        checkMap
        auto x = AstNodeImpl::asExpressionImpl(clone(node.arg()));

        AllReduceImpl* a;
         if (x->type() == TensorNode) {
            a = new AllReduceImpl(AstNodeImpl::asTensorImpl(x), node.reduceOp());
        } else {
            a = new AllReduceImpl(AstNodeImpl::asStageImpl(x), node.reduceOp());
        }
        addToMap(node, a);
    }
    virtual void visit(SendImpl& node)
    {
        checkMap
        auto x = AstNodeImpl::asExpressionImpl(clone(node.arg()));
        auto y = AstNodeImpl::asProcessGroupIDImpl(clone(node.dstGroup()));
        auto r = AstNodeImpl::asExpressionImpl(clone(node.dstRank()));

        SendImpl* a;
         if (x->type() == TensorNode) {
            a = new SendImpl(AstNodeImpl::asTensorImpl(x), y, r);
        } else {
            a = new SendImpl(AstNodeImpl::asStageImpl(x), y, r);
        }
        addToMap(node, a);
    }
    virtual void visit(ReduceImpl& node)
    {
        ASSERT(false, "to implement");
        // checkMap
        // ReduceImpl* a = new ReduceImpl(clone(node.arg()), node.root());
        // addToMap(node, a);
    }
    virtual void visit(NormImpl& node)
    {
        checkMap
        auto cloneArg = clone(node.arg());
        NormImpl* a;
        if (cloneArg->type() == TensorNode)
            a = new NormImpl(AstNodeImpl::asTensorImpl(cloneArg));
        else 
            a = new NormImpl(AstNodeImpl::asStageImpl(cloneArg));
        addToMap(node, a);
    }
    virtual void visit(BroadcastImpl& node)
    {
        ASSERT(false, "to implement");
        // checkMap
        // BroadcastImpl* a = new BroadcastImpl(clone(node.arg()), node.root());
        // addToMap(node, a);
    }
    virtual void visit(AllGatherImpl& node)
    {
        checkMap
        auto x = AstNodeImpl::asExpressionImpl(clone(node.arg()));

        AllGatherImpl* a;
         if (x->type() == TensorNode) {
            a = new AllGatherImpl(AstNodeImpl::asTensorImpl(x));
        } else {
            a = new AllGatherImpl(AstNodeImpl::asStageImpl(x));
        }        
        addToMap(node, a);
    }
    virtual void visit(ReduceScatterImpl& node)
    {
        checkMap
        auto x = AstNodeImpl::asExpressionImpl(clone(node.arg()));

        ReduceScatterImpl* a;
         if (x->type() == TensorNode) {
            a = new ReduceScatterImpl(AstNodeImpl::asTensorImpl(x), node.reduceOp());
        } else {
            a = new ReduceScatterImpl(AstNodeImpl::asStageImpl(x), node.reduceOp());
        }        
        addToMap(node, a);
    }
    virtual void visit(BinaryPointwiseOp& node)
    {
        checkMap
        auto op0 = clone(node.operand(0));
        auto op1 = clone(node.operand(1));
        BinaryPointwiseOp* b = new BinaryPointwiseOp(node.op(), AstNodeImpl::asExpressionImpl(op0), 
                                                     AstNodeImpl::asExpressionImpl(op1));
        addToMap(node, b);
    }
    virtual void visit(MatMulImpl& node)
    {
        checkMap
        auto op0 = AstNodeImpl::asExpressionImpl(clone(node.operand(0)));
        auto op1 = AstNodeImpl::asExpressionImpl(clone(node.operand(1)));
        MatMulImpl* b = new MatMulImpl(op0, op1);
        addToMap(node, b);
    }
    virtual void visit(UnaryPointwiseOp& node)
    {
        checkMap
        auto op = clone(node.operand());
        UnaryPointwiseOp* b = new UnaryPointwiseOp(node.op(), 
            AstNodeImpl::asExpressionImpl(op));
        addToMap(node, b);
    }
    virtual void visit(PowerImpl& node)
    {
        checkMap
        PowerImpl* b = new PowerImpl(AstNodeImpl::asExpressionImpl(clone(node.operand())), node.n());
        addToMap(node, b);
    }
    virtual void visit(ReduceTensorImpl& node)
    {
        checkMap
        auto x = AstNodeImpl::asExpressionImpl(clone(node.arg()));

        ReduceTensorImpl* a;
         if (x->type() == TensorNode) {
            a = new ReduceTensorImpl(AstNodeImpl::asTensorImpl(x), node.op());
        } else {
            a = new ReduceTensorImpl(AstNodeImpl::asStageImpl(x), node.op());
        }
        addToMap(node, a);
    }
    virtual void visit(ProcessGroupImpl& node) {
        checkMap
        ProcessGroupImpl* b = new ProcessGroupImpl(AstNodeImpl::asProcessGroupImpl(clone(node.parent())), 
                                                  AstNodeImpl::asExpressionImpl(clone(node.splitSize())));
        addToMap(node, b);
    }
    virtual void visit(ProcessGroupIDImpl& node) {
        checkMap
        ProcessGroupIDImpl* b = new ProcessGroupIDImpl(AstNodeImpl::asProcessGroupImpl(clone(node.group())), 
                                                       node.idType());
        addToMap(node, b);
    }
    virtual void visit(StageImpl& node)
    {
        checkMap
        StageImpl* b = new StageImpl(AstNodeImpl::asExpressionImpl(clone(node.definition())));
        addToMap(node, b);
    }
    virtual void visit(DropoutImpl& node)
    {
        checkMap
        DropoutImpl* b = new DropoutImpl(AstNodeImpl::asStageImpl(clone(node.arg())), node.prob());
        addToMap(node, b);
    }
    virtual void visit(UpdateImpl& node)
    {
        checkMap
        UpdateImpl* b = new UpdateImpl(AstNodeImpl::asTensorImpl(clone(node.arg())), 
                                       AstNodeImpl::asExpressionImpl(clone(node.update())));
        addToMap(node, b);
    }
    virtual void visit(VariableImpl& node)
    {
        VariableImpl* v = new VariableImpl(node);
        addToMap(node, v);
    }
    virtual void visit(CastImpl& node)
    {
        checkMap
        CastImpl* b = new CastImpl(node.elemType(), AstNodeImpl::asExpressionImpl(clone(node.op())));
        addToMap(node, b);
    }
    virtual void visit(ConstUInt64& node)
    {
        checkMap
        ConstUInt64* b = new ConstUInt64(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstInt64& node)
    {
        checkMap
        ConstInt64* b = new ConstInt64(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstUInt32& node)
    {
        checkMap
        ConstUInt32* b = new ConstUInt32(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstInt32& node)
    {
        checkMap
        ConstInt32* b = new ConstInt32(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstFloat16& node)
    {
        checkMap
        ConstFloat16* b = new ConstFloat16(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstFloat32& node)
    {
        checkMap
        ConstFloat32* b = new ConstFloat32(node.val());
        addToMap(node, b);
    }
    virtual void visit(ConstFloat64& node)
    {
        checkMap
        ConstFloat64* b = new ConstFloat64(node.val());
        addToMap(node, b);
    }
    virtual void visit(ScatterImpl& node)
    {
        checkMap
        ScatterImpl* b;
        auto argClone = clone(node.arg());
        if (argClone->type() == TensorNode) {
            b = new ScatterImpl(AstNodeImpl::asTensorImpl(argClone));
        } else {
            b = new ScatterImpl(AstNodeImpl::asStageImpl(argClone));
        }
        addToMap(node, b);
    }
    virtual void visit(IteImpl& node)
    {
        checkMap
        IteImpl* b = new IteImpl(AstNodeImpl::asExpressionImpl(clone(node.cond())),
                                 AstNodeImpl::asExpressionImpl(clone(node.ifTrue())),
                                 AstNodeImpl::asExpressionImpl(clone(node.ifFalse())));
        addToMap(node, b);
    }
};

class VisitChildrenVisitor : public AstVisitor {
    public: 
    VisitChildrenVisitor() {}
    
    virtual void visit(TensorImpl& node) {
        visitChildren(node);
    }
    virtual void visit(AllReduceImpl& node) {
        visitChildren(node);
    }

    virtual void visit(ReduceImpl& node) 
    {
        visitChildren(node);
    }

    virtual void visit(BroadcastImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(AllGatherImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(ReduceScatterImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(SendImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(BinaryPointwiseOp& node) {
        node.operand(0)->accept(*this);
        node.operand(1)->accept(*this);
    }
    virtual void visit(MatMulImpl& node) {
        node.operand(0)->accept(*this);
        node.operand(1)->accept(*this);
    }
    virtual void visit(UnaryPointwiseOp& node) {
        visitChildren(node);
    }
    virtual void visit(DropoutImpl& node)
    {
        visitChildren(node);
    }
    virtual void visit(PowerImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ReduceTensorImpl& node) {
        visitChildren(node);
    }
    virtual void visit(NormImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ProcessGroupImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ProcessGroupIDImpl& node) {
        visitChildren(node);
    }
    virtual void visit(CastImpl& node) {
        visitChildren(node);
    }
    virtual void visit(UpdateImpl& node) {
        visitChildren(node);
    }
    virtual void visit(StageImpl& node) {
    }
    
    virtual void visit(ScatterImpl& node) {
        visitChildren(node);
    }
    
    virtual void visit(IteImpl& node) {
        visitChildren(node);
    }

    virtual void visit(VariableImpl& node) {
    }

    virtual void visit(ConstUInt64& node) {
    }
    virtual void visit(ConstInt64& node) {
    }
    virtual void visit(ConstUInt32& node) {

    }
    virtual void visit(ConstInt32& node) {

    }
    virtual void visit(ConstFloat16& node){}

    virtual void visit(ConstFloat32& node) {
    }
    virtual void visit(ConstFloat64& node) {
    }
};

class StageParentsVisitor : public AstVisitor {
private:
    std::vector<StageImpl*> parents;
    
public:
    StageParentsVisitor() {}
    
    std::vector<StageImpl*>& getParents() {return parents;}
    void visit(TensorImpl& node) {
        visitChildren(node);
    }
    void visit(AllReduceImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ReduceImpl& node) {
        visitChildren(node);
    }
    virtual void visit(BroadcastImpl& node) {
        visitChildren(node);
    }
    virtual void visit(AllGatherImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ReduceScatterImpl& node) {
        visitChildren(node);
    }
    virtual void visit(SendImpl& node) {
        visitChildren(node);
    }
    virtual void visit(NormImpl& node) {
        visitChildren(node);
    }
    void visit(BinaryPointwiseOp& node) {
        node.operand(0)->accept(*this);
        node.operand(1)->accept(*this);
    }
    void visit(MatMulImpl& node) {
        node.operand(0)->accept(*this);
        node.operand(1)->accept(*this);
    }
    void visit(UnaryPointwiseOp& node) {
        visitChildren(node);
    }
    void visit(PowerImpl& node) {
        visitChildren(node);
    }
    void visit(ReduceTensorImpl& node) {
        visitChildren(node);
    }
    void visit(ProcessGroupImpl& node) {
        //ProcessGroup has no parents
    }
    void visit(ProcessGroupIDImpl& node) {
        //ProcessGroup has no parents
    }
    void visit(CastImpl& node) {
        visitChildren(node);
    }
    void visit(StageImpl& node) {
        parents.push_back(&node);
    }
    void visit(UpdateImpl& node) {
        visitChildren(node);
    }
    void visit(VariableImpl& node) {
    }

    void start(StageImpl& node) {
        visitChildren(node);
    }
    virtual void visit(DropoutImpl& node)
    {
        visitChildren(node);
    }
    virtual void visit(ScatterImpl& node) {
        visitChildren(node);
    }

    virtual void visit(IteImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ConstUInt64& node) {}
    virtual void visit(ConstInt64& node) {}
    virtual void visit(ConstUInt32& node) {}
    virtual void visit(ConstInt32& node) {}
    virtual void visit(ConstFloat32& node) {}
    virtual void visit(ConstFloat64& node) {}
    virtual void visit(ConstFloat16& node) {}
};

class SingleStagePrintVisitor : public AstVisitor {
private:
    std::ostream& os_;
    int nameCounter;
public:
    SingleStagePrintVisitor(std::ostream& os) : os_(os), nameCounter(0) {}
    
    void visit(TensorImpl& node) {
        os_ << node.name();
        visitChildren(node);
    }
    void visit(AllReduceImpl& node) {
        os_ << "allreduce" << "(";
        visitChildren(node);
        os_ << ")";
    }

    virtual void visit(ReduceImpl& node) 
    {
        os_ << "reduce" << "(";
        visitChildren(node);
        os_ << ")";
    }

    virtual void visit(BroadcastImpl& node) 
    {
        os_ << "broadcast"<<"(";
        visitChildren(node);
        os_ << ")";
    }
    virtual void visit(AllGatherImpl& node) 
    {
        os_ << "allgather"<<"(";
        visitChildren(node);
        os_ << ")";
    }
    virtual void visit(ReduceScatterImpl& node) 
    {
        os_ << "reduce-scatter"<<"(";
        visitChildren(node);
        os_ << ")";
    }
    virtual void visit(SendImpl& node) {
        os_ << "send(";
        node.arg()->accept(*this);
        os_ << ", ";
        node.dstGroup()->accept(*this);
        os_ << ", ";
        node.dstRank()->accept(*this);
        os_ << ")";
    }
    void visit(BinaryPointwiseOp& node) {
        os_ << "(";
        node.operand(0)->accept(*this);
        os_ << " " << BinaryPointwiseOp::operatorToStr(node.op()) << " ";
        node.operand(1)->accept(*this);
        os_ << ")";
    }
    void visit(MatMulImpl& node) {
        os_ << "MatMul(";
        node.operand(0)->accept(*this);
        os_ << ",";
        node.operand(1)->accept(*this);
        os_ << ")";
    }
    void visit(DropoutImpl& node) {
        os_ << "Dropout(";
        node.arg()->accept(*this);
        os_ << ",";
        os_ << node.prob();
        os_ << ")";
    }
    void visit(ProcessGroupImpl& node) {
        node.parent()->accept(*this);
    }
    void visit(ProcessGroupIDImpl& node) {
        node.group()->accept(*this);
        if (node.idType() == NextProcessGroupID)
            os_ << " + 1";
        else if (node.idType() == PreviousProcessGroupID)
            os_ << " - 1";
    }
    void visit(UnaryPointwiseOp& node) {
        visitChildren(node);
    }
    void visit(PowerImpl& node) {
        visitChildren(node);
    }
    void visit(ReduceTensorImpl& node) {
        visitChildren(node);
    }
    void visit(NormImpl& node) {
        os_<<"Norm(";
        node.arg()->accept(*this);
        os_ << ")";
    }
    void visit(StageImpl& node) {
        os_ << node.name();
    }
    void visit(UpdateImpl& node) {
        os_ << "Update(";
        node.arg()->accept(*this);
        os_ << ", ";
        node.update()->accept(*this);
        os_ << ")";
    }
    virtual void visit(CastImpl& node) {
        os_ << node.name() << "(" << elemTypeToCType(node.elemType());
        visitChildren(node);
        os_ << node.name() << ")";
    }

    void print(StageImpl& node) {
         os_ << node.name() << " = ";
         visitChildren(node);
         os_ << "\n";
    }

    virtual void visit(ScatterImpl& node) {
        os_ << "slice (" ;
        visitChildren(node);
        os_ << ")";
    }

    virtual void visit(IteImpl& node) {
        os_ << "(";
        node.cond()->accept(*this);
        os_ << " ? ";
        node.ifTrue()->accept(*this);
        os_ << " : ";
        node.ifFalse()->accept(*this);
        os_ << ")";
        visitChildren(node);
    }

    void visit(VariableImpl& node) {
        os_ << node.name();
    }

    virtual void visit(ConstUInt64& node) {
        os_ << node.val();
    }
    virtual void visit(ConstInt64& node) {
        os_ << node.val();
    }
    virtual void visit(ConstUInt32& node) {
        os_ << node.val();
    }
    virtual void visit(ConstInt32& node) {
        os_ << node.val();
    }
    virtual void visit(ConstFloat16& node) {
        os_ << "half(" << node.val() << ")";
    }
    virtual void visit(ConstFloat32& node) {
        os_ << node.val();
    }
    virtual void visit(ConstFloat64& node) {
        os_ << node.val();
    }
};

class InputsVisitor : public VisitChildrenVisitor {
private:
    std::set<ExpressionImpl*> inputs_;
public:
    InputsVisitor() {}
    
    virtual void visit(TensorImpl& node) {
        inputs_.insert(&node);
    }

    virtual void visit(StageImpl& node) {
        inputs_.insert(&node);
    }

    virtual void visit(VariableImpl& node) {
        inputs_.insert(&node);
    }

    std::set<ExpressionImpl*> inputs(StageImpl& node) {
        visitChildren(node);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(BinaryPointwiseOp& node) {
        node.accept(*this);
        return inputs_;
    }


    std::set<ExpressionImpl*> inputs(MatMulImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(UnaryPointwiseOp& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(IteImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(AllReduceImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(ReduceScatterImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(AllGatherImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(ExpressionImpl& node) {
        node.accept(*this);
        return inputs_;
    }

    std::set<ExpressionImpl*> inputs(NormImpl& node) {
        node.accept(*this);
        return inputs_;
    }
};

class AllCastOpsVisitor : public VisitChildrenVisitor
{
private:
    std::vector<CastImpl*> castOps_;
public:
    AllCastOpsVisitor() : castOps_()
    {

    }

    virtual void visit(CastImpl& node) {
        castOps_.push_back(&node);
    }

    std::vector<CastImpl*> castOps (ExpressionImpl& node)
    {
        node.accept(*this);
        return castOps_;
    }
};


class ReplaceExprVisitor : public VisitChildrenVisitor {
private:
    ExpressionImpl* origExpr;
    ExpressionImpl* replacementExpr;

public:
    ReplaceExprVisitor(ExpressionImpl* _origExpr, ExpressionImpl* _replacementExpr):
        origExpr(_origExpr), replacementExpr(_replacementExpr) {}
    
    virtual void visit(AllReduceImpl& node) {
        if (node.arg().get() == origExpr) {
            
        }
    }

    virtual void visit(ReduceImpl& node) 
    {
        visitChildren(node);
    }

    virtual void visit(BroadcastImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(AllGatherImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(ReduceScatterImpl& node) 
    {
        visitChildren(node);
    }
    virtual void visit(BinaryPointwiseOp& node) {
        node.operand(0)->accept(*this);
        node.operand(1)->accept(*this);
    }
    virtual void visit(UnaryPointwiseOp& node) {
        visitChildren(node);
    }
    virtual void visit(PowerImpl& node) {
        visitChildren(node);
    }
    virtual void visit(ReduceTensorImpl& node) {
        visitChildren(node);
    }
    virtual void visit(CastImpl& node) {
        visitChildren(node);
    }
    virtual void visit(StageImpl& node) {
    }
    
    virtual void visit(ScatterImpl& node) {
        visitChildren(node);
    }
    
    virtual void visit(IteImpl& node) {
        visitChildren(node);
    }    
};
}

#endif
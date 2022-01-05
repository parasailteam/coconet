#ifndef __DSL_HPP__
#define __DSL_HPP__

#include "ast.hpp"
#include <vector>
#include <memory>
#include <type_traits>

#define declImpl(x) \
    const std::shared_ptr<ACCCDSLImpl::x##Impl> impl() const {return ACCCDSLImpl::AstNodeImpl::as##x##Impl(exprImpl_);}

namespace ACCCDSL {
class Stage;
class Tensor;

class Expression {
protected:
    std::shared_ptr<ACCCDSLImpl::ExpressionImpl> exprImpl_;
    Expression(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> exprImpl) : exprImpl_(exprImpl) {}
    Expression(ACCCDSLImpl::ExpressionImpl* exprImpl) : exprImpl_(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> (exprImpl)) {}

public:
    const std::shared_ptr<ACCCDSLImpl::ExpressionImpl> impl() const {return exprImpl_;}
    Expression(const Expression& x) 
    {
        exprImpl_ = x.impl();
    }
};

class SingleDimExpression : public Expression {
public:
    SingleDimExpression(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> exprImpl) : Expression(exprImpl) {}
    SingleDimExpression(ACCCDSLImpl::ExpressionImpl* exprImpl) : Expression(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> (exprImpl)) {}
    const std::shared_ptr<ACCCDSLImpl::ExpressionImpl> impl() const {return exprImpl_;}
};

class ContinuousExpression : public Expression {
    public:
    ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> exprImpl) : Expression(exprImpl) {}
    ContinuousExpression(ACCCDSLImpl::ExpressionImpl* exprImpl) : Expression(exprImpl) {}
    
    template<ACCCDSLImpl::BinaryOp op>
    ContinuousExpression genericOperatorOverload(ContinuousExpression& x, ContinuousExpression& y)
    {
        auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, x.impl(), y.impl(), false);
        return ContinuousExpression(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
    }

    virtual ContinuousExpression operator-(ContinuousExpression y)
    {
       return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Subtract>(*this, y);
    }

    virtual ContinuousExpression operator*(ContinuousExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Multiply>(*this, y);
    }

    virtual ContinuousExpression operator+(ContinuousExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Add>(*this, y);
    }

    virtual ContinuousExpression operator/(ContinuousExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Divide>(*this, y);
    }
};

/*Operators with left operand as constant and right as ContinuousExpression*/
ContinuousExpression operator-(float v, ContinuousExpression x);
ContinuousExpression operator+(float v, ContinuousExpression x);
ContinuousExpression operator*(float v, ContinuousExpression x);
ContinuousExpression operator/(float v, ContinuousExpression x);

/*Operators with left operand as ContinuousExpression and right as constant*/
ContinuousExpression operator-(ContinuousExpression x, float v);

ContinuousExpression operator+(ContinuousExpression x, float v);

ContinuousExpression operator*(ContinuousExpression x, float v);

ContinuousExpression operator/(ContinuousExpression x, float v);

/*Operators with left operand as ContinuousExpression and right as constant*/
ContinuousExpression operator-(ContinuousExpression x, SingleDimExpression v);
ContinuousExpression operator+(ContinuousExpression x, SingleDimExpression v);
ContinuousExpression operator*(ContinuousExpression x, SingleDimExpression v);
ContinuousExpression operator/(ContinuousExpression x, SingleDimExpression v);
ContinuousExpression operator-(SingleDimExpression v, ContinuousExpression x);
ContinuousExpression operator+(SingleDimExpression v, ContinuousExpression x);
ContinuousExpression operator*(SingleDimExpression v, ContinuousExpression x);
ContinuousExpression operator/(SingleDimExpression v, ContinuousExpression x);

//TODO define above operators for all types

class Variable : public SingleDimExpression 
{
public:
    Variable(TensorElemType t, std::string name) : SingleDimExpression(std::shared_ptr<ACCCDSLImpl::VariableImpl>(new ACCCDSLImpl::VariableImpl(t, name)))
    {}

    std::shared_ptr<ACCCDSLImpl::VariableImpl> impl() {return ACCCDSLImpl::AstNodeImpl::asVariableImpl(exprImpl_);}
};

class Tensor : public ContinuousExpression {
public:
    Tensor(std::shared_ptr<ACCCDSLImpl::TensorImpl> impl) : ContinuousExpression(impl) {}
    //FIXME: above constructor should only be available to pipeline
    // friend std::pair<ACCCDSL::ScatteredStage, ACCCDSL::Stage> ACCCDSL::Pipeline::split(Stage& stageToSplit, SplitType splitType);

public:
    Tensor(TensorElemType elemType, Variable n1, Variable n2, TensorLayout layout, std::string name) :
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, n1.impl(), n2.impl(), layout, name, false)))
    {}

    Tensor(TensorElemType elemType, Variable n, TensorLayout layout, std::string name) :
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, n.impl(), layout, name, false)))
    {}

    Tensor(TensorElemType elemType, std::vector<Variable> dimSizes, TensorLayout layout, std::string name) :
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, std::vector<std::shared_ptr<ACCCDSLImpl::ExpressionImpl>>(0, nullptr), layout, name, false)))
    {
        for (auto d : dimSizes) {
            impl()->addDim(d.impl());
        }
    }

    declImpl(Tensor)
};

class Stage : public ContinuousExpression {
public:
    Stage(std::shared_ptr<ACCCDSLImpl::StageImpl> stageImpl) : 
          ContinuousExpression(stageImpl)
    {
        
    }
public:
    Stage(ContinuousExpression definition) : 
          ContinuousExpression(std::shared_ptr<ACCCDSLImpl::StageImpl>(new ACCCDSLImpl::StageImpl(definition.impl(), false)))
    {
        
    }

    declImpl(Stage)
};

class Dropout : public ContinuousExpression {
public:
    Dropout(Stage expr, float prob) : 
          ContinuousExpression(std::shared_ptr<ACCCDSLImpl::DropoutImpl>(new ACCCDSLImpl::DropoutImpl(expr.impl(), prob)))
    {
        
    }

    declImpl(Dropout)
};

class Update : public ContinuousExpression {
public:
    Update(Tensor t, ContinuousExpression definition) : 
          ContinuousExpression(std::shared_ptr<ACCCDSLImpl::UpdateImpl>(new ACCCDSLImpl::UpdateImpl(ACCCDSLImpl::AstNodeImpl::asTensorImpl(t.impl()), definition.impl())))
    {
        
    }
};


/*Scattered Expression*/
class ScatteredExpression : public Expression {
    public:
    ScatteredExpression(std::shared_ptr<ACCCDSLImpl::ExpressionImpl> exprImpl) : Expression(exprImpl) {}
    ScatteredExpression(ACCCDSLImpl::ExpressionImpl* exprImpl) : Expression(exprImpl) {}
    
    template<ACCCDSLImpl::BinaryOp op>
    ScatteredExpression genericOperatorOverload(ScatteredExpression& x, ScatteredExpression& y)
    {
        auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, x.impl(), y.impl(), true);
        return ScatteredExpression(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
    }

    virtual ScatteredExpression operator-(ScatteredExpression y)
    {
       return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Subtract>(*this, y);
    }

    virtual ScatteredExpression operator*(ScatteredExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Multiply>(*this, y);
    }

    virtual ScatteredExpression operator+(ScatteredExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Add>(*this, y);
    }

    virtual ScatteredExpression operator/(ScatteredExpression y)
    {
        return genericOperatorOverload<ACCCDSLImpl::BinaryOp::Divide>(*this, y);
    }
};

ScatteredExpression operator-(float v, ScatteredExpression x);
ScatteredExpression operator+(float v, ScatteredExpression x);
ScatteredExpression operator*(float v, ScatteredExpression x);
ScatteredExpression operator/(float v, ScatteredExpression x);

/*Operators with left operand as ContinuousExpression and right as constant*/
ScatteredExpression operator-(ScatteredExpression x, float v);
ScatteredExpression operator+(ScatteredExpression x, float v);
ScatteredExpression operator*(ScatteredExpression x, float v);
ScatteredExpression operator/(ScatteredExpression x, float v);

ScatteredExpression operator-(ScatteredExpression x, SingleDimExpression v);
ScatteredExpression operator*(ScatteredExpression x, SingleDimExpression v);
ScatteredExpression operator/(ScatteredExpression x, SingleDimExpression v);
ScatteredExpression operator+(ScatteredExpression x, SingleDimExpression v);

ScatteredExpression operator-(SingleDimExpression x, ScatteredExpression v);
ScatteredExpression operator+(SingleDimExpression x, ScatteredExpression v);
ScatteredExpression operator/(SingleDimExpression x, ScatteredExpression v);
ScatteredExpression operator*(SingleDimExpression x, ScatteredExpression v);

SingleDimExpression operator-(SingleDimExpression x, SingleDimExpression v);
SingleDimExpression operator*(SingleDimExpression x, SingleDimExpression v);
SingleDimExpression operator/(SingleDimExpression x, SingleDimExpression v);
SingleDimExpression operator+(SingleDimExpression x, SingleDimExpression v);

SingleDimExpression operator-(float x, SingleDimExpression v);
SingleDimExpression operator*(float x, SingleDimExpression v);
SingleDimExpression operator/(float x, SingleDimExpression v);
SingleDimExpression operator+(float x, SingleDimExpression v);

SingleDimExpression operator-(SingleDimExpression x, float v);
SingleDimExpression operator*(SingleDimExpression x, float v);
SingleDimExpression operator/(SingleDimExpression x, float v);
SingleDimExpression operator+(SingleDimExpression x, float v);

ContinuousExpression operator>(ContinuousExpression x, float v);
SingleDimExpression operator>(SingleDimExpression x, float v);

class Sqrt : public ContinuousExpression {
public:
    Sqrt(ContinuousExpression expr) : ContinuousExpression(std::shared_ptr<ACCCDSLImpl::UnaryPointwiseOp>(
        new ACCCDSLImpl::UnaryPointwiseOp(ACCCDSLImpl::SqrtOp, expr.impl()))) {}
};

class MatMul : public ContinuousExpression {
public:
    MatMul(Tensor m1, Tensor m2) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::MatMulImpl>(
            new ACCCDSLImpl::MatMulImpl(ACCCDSLImpl::AstNodeImpl::asTensorImpl(m1.impl()),
                                        ACCCDSLImpl::AstNodeImpl::asTensorImpl(m2.impl())))) {}
    MatMul(Tensor m1, Stage m2) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::MatMulImpl>(
            new ACCCDSLImpl::MatMulImpl(ACCCDSLImpl::AstNodeImpl::asTensorImpl(m1.impl()),
                                        ACCCDSLImpl::AstNodeImpl::asStageImpl(m2.impl())))) {}
    MatMul(Stage m1, Tensor m2) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::MatMulImpl>(
            new ACCCDSLImpl::MatMulImpl(ACCCDSLImpl::AstNodeImpl::asStageImpl(m1.impl()),
                                        ACCCDSLImpl::AstNodeImpl::asTensorImpl(m2.impl())))) {}
    MatMul(Stage m1, Stage m2) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::MatMulImpl>(
            new ACCCDSLImpl::MatMulImpl(ACCCDSLImpl::AstNodeImpl::asStageImpl(m1.impl()),
                                        ACCCDSLImpl::AstNodeImpl::asStageImpl(m2.impl())))) {}
};

class ReduceStage : public ContinuousExpression {
public:
    ReduceStage(Stage definition, ReduceOperation op) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ReduceTensorImpl>(new ACCCDSLImpl::ReduceTensorImpl(definition.impl(), op)))
    {
    }
};

class Norm : public ContinuousExpression {
public:
    Norm(Stage in) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::NormImpl>(new ACCCDSLImpl::NormImpl(in.impl()))) {}
    Norm(Tensor in) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::NormImpl>(new ACCCDSLImpl::NormImpl(in.impl()))) {}
};

// class ScatteredTensor : public ScatteredExpression {
// public:
//     ScatteredTensor(TensorElemType elemType, size_t n1, size_t n2, std::string name) :
//         ScatteredExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, n1, n2, Sliced, name, true)))
//     {}

//     ScatteredTensor(TensorElemType elemType, size_t n, std::string name) :
//         ScatteredExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, n, Sliced, name, true)))
//     {}

//     ScatteredTensor(TensorElemType elemType, std::vector<size_t>& dimSizes, std::string name) :
//         ScatteredExpression(std::shared_ptr<ACCCDSLImpl::TensorImpl>(new ACCCDSLImpl::TensorImpl(elemType, dimSizes, Sliced, name, true)))
//     {}
// };

// class ScatteredStage : public ScatteredExpression {
// public:
//     ScatteredStage(ScatteredExpression definition) : 
//         ScatteredExpression(std::shared_ptr<ACCCDSLImpl::StageImpl>(new ACCCDSLImpl::StageImpl(definition.impl(), true)))
//     {
//     }
// };

class Cast : public ContinuousExpression {
public:
    Cast(TensorElemType elemType, ContinuousExpression op) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::CastImpl>(new ACCCDSLImpl::CastImpl(elemType, op.impl())))
    {}
};

template<class T>
class Const : public SingleDimExpression {
public:
    Const(TensorElemType t, T val) :
        SingleDimExpression(ACCCDSLImpl::constantValToConstantImpl(t, val))
    {}
};

class ScatteredCast : public ScatteredExpression {
public:
    ScatteredCast(TensorElemType elemType, ScatteredExpression op) : 
        ScatteredExpression(std::shared_ptr<ACCCDSLImpl::CastImpl>(new ACCCDSLImpl::CastImpl(elemType, op.impl())))
    {}
};

/*Collective Communications*/
class AllReduce_ : public ContinuousExpression {
public:
    AllReduce_(ReduceOperation op, Tensor& t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::AllReduceImpl>(new ACCCDSLImpl::AllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl()), op))) {}
    
    AllReduce_(ReduceOperation op, Stage& s) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::AllReduceImpl>(new ACCCDSLImpl::AllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl()), op))) {}
};

class AllGather_ : public ContinuousExpression {
public:
    AllGather_(std::shared_ptr<ACCCDSLImpl::StageImpl> t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::AllGatherImpl>(new ACCCDSLImpl::AllGatherImpl((t)))) {}
public:
    AllGather_(Stage& t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::AllGatherImpl>(new ACCCDSLImpl::AllGatherImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(t.impl())))) {}
};

class ReduceScatter_ : public ContinuousExpression {
public:
    ReduceScatter_(ReduceOperation op, Tensor t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ReduceScatterImpl>(new ACCCDSLImpl::ReduceScatterImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl()), op))) {}
    
    ReduceScatter_(ReduceOperation op, Stage s) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ReduceScatterImpl>(new ACCCDSLImpl::ReduceScatterImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl()), op))) {}
    
};

class Scatter_ : public ContinuousExpression {
public:
    Scatter_(std::shared_ptr<ACCCDSLImpl::StageImpl> expr) : ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ScatterImpl>(new ACCCDSLImpl::ScatterImpl(expr))) {}
    
    Scatter_(std::shared_ptr<ACCCDSLImpl::TensorImpl> expr) : ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ScatterImpl>(new ACCCDSLImpl::ScatterImpl(expr))) {}

public:
    Scatter_(Tensor t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ScatterImpl>(new ACCCDSLImpl::ScatterImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl())))) {}
    
    Scatter_(Stage s) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::ScatterImpl>(new ACCCDSLImpl::ScatterImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl())))) {}
};

class Broadcast_ : public ContinuousExpression {
public:
    Broadcast_(Tensor& t, int NumGPUs) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::BroadcastImpl>(new ACCCDSLImpl::BroadcastImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl()), NumGPUs))) {}
    
    Broadcast_(Stage& s, int NumGPUs) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::BroadcastImpl>(new ACCCDSLImpl::BroadcastImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl()), NumGPUs))) {}
};

class Send_ : public ContinuousExpression {
public:
    Send_(Tensor& t, Variable dest) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::SendImpl>(new ACCCDSLImpl::SendImpl(t.impl(), dest.impl()))) {}
    
    Send_(Stage& s, Variable dest) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::SendImpl>(new ACCCDSLImpl::SendImpl(s.impl(), dest.impl()))) {}
};

// class Reduce_ : public Expression {
// public:
//     Reduce_(ReduceOperation op, Tensor& t) : 
//         Expression(std::shared_ptr<ACCCDSLImpl::AllReduceImpl>(new ACCCDSLImpl::AllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl()), op))) {}
    
//     Reduce_(ReduceOperation op, Stage& s) : 
//         Expression(std::shared_ptr<ACCCDSLImpl::AllReduceImpl>(new ACCCDSLImpl::AllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl()), op))) {}
// };

enum CollCommOperationType {
    NoneCollCommOp,
    AllReduceOp,
    AllGatherOp,
    ReduceOp,
    ReduceScatterOp,
    BroadcastOp,
    ScatterOp,
    SendOp
};

template<class T>
class CollCommOperation {
private:
    CollCommOperationType op_;

public:
    CollCommOperation(CollCommOperationType op) : op_(op) {}

    T operator()(ReduceOperation reduceOp, Tensor t) {
        return T(reduceOp, t);
    }

    T operator()(ReduceOperation reduceOp, Stage s) {
        return T(reduceOp, s);
    }

    T operator()(Tensor& s) {
        return T(s);
    }

    T operator()(Stage& s) {
        return T(s);
    }

    T operator()(Tensor& t, Variable dst) {
        return T(t, dst);
    }

    T operator()(Stage& s, Variable dst) {
        return T(s, dst);
    }
};

extern CollCommOperation<AllReduce_> AllReduce;
extern CollCommOperation<AllGather_> AllGather;
extern CollCommOperation<ReduceScatter_> ReduceScatter;
extern CollCommOperation<Broadcast_> Broadcast;
extern CollCommOperation<Scatter_> Scatter;
extern CollCommOperation<Send_> Send;

//TODO: Supporting Fused Communication Collectives is not a priority right now
class FusedAllReduce_ : public ContinuousExpression {
protected:
    //A FusedAllReduce always need Sliced stage
    Stage* fusedComp;

public:
    FusedAllReduce_(ReduceOperation op, Tensor& t) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::FusedAllReduceImpl>(new ACCCDSLImpl::FusedAllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::TensorImpl>(t.impl()), op))) {}
    
    FusedAllReduce_(ReduceOperation op, Stage& s) : 
        ContinuousExpression(std::shared_ptr<ACCCDSLImpl::FusedAllReduceImpl>(new ACCCDSLImpl::FusedAllReduceImpl(std::dynamic_pointer_cast<ACCCDSLImpl::StageImpl>(s.impl()), op))) {}
    
    void setFusedComp(Stage& stage) {
        fusedComp = &stage;
    }
};

template<class T, class U>
class FusedCollComm {  
public:
    FusedCollComm(T& commCollStage, U& fusedComStage) {}

    T operator()(ReduceOperation reduceOp, Tensor& t) {
        return T(reduceOp, t);
    }

    T operator()(ReduceOperation reduceOp, Stage& s) {
        return T(reduceOp, s);
    }

    T operator()(Tensor& s) {
        return T(s);
    }

    T operator()(Stage& s) {
        return T(s);
    }
};

//TODO: Currently only supports FusedAllReduce
// extern FusedCollCommOperation<FusedAllReduce_> FusedAllReduce;

// extern FusedCollCommOperation<AllGather_> FusedAllGather;
// extern FusedCollCommOperation<ReduceScatter_> FusedReduceScatter;
// extern FusedCollCommOperation<Broadcast_> FusedBroadcast;
// extern FusedCollCommOperation<Scatter_> FusedScatter;
// CollCommOperation<Reduce_> Reduce(ReduceOp);

ContinuousExpression Sqrt(ContinuousExpression x);

ContinuousExpression Ite(ContinuousExpression cond, ContinuousExpression ifTrue, ContinuousExpression ifFalse);

ContinuousExpression Ite(ContinuousExpression cond, ContinuousExpression ifTrue, float ifFalse);

ContinuousExpression Ite(SingleDimExpression cond, ContinuousExpression ifTrue, float ifFalse);

}

#include "keywords.hpp"

#endif
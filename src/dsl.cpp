#include <dsl.hpp>

using namespace ACCCDSL;

template<class T> 
std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::_constantValToConstantImpl(TensorElemType t, T val)
{
    switch(t) {
        case Float16:
        {
            return ACCCDSLImpl::constantValToConstantImpl((float)val, true);
        }
        case Float32:
        {
            return ACCCDSLImpl::constantValToConstantImpl((float)val, false);
        }
        case Float64:
        {
            return ACCCDSLImpl::constantValToConstantImpl((double)val);
        }
        case Int32:
        {
            return ACCCDSLImpl::constantValToConstantImpl((int32_t)val);
        }
        case Int64:
        {
            return ACCCDSLImpl::constantValToConstantImpl((int64_t)val);
        }
        case UInt32:
        {
            return ACCCDSLImpl::constantValToConstantImpl((uint32_t)val);
        }
        case UInt64:
        {
            return ACCCDSLImpl::constantValToConstantImpl((uint64_t)val);
        }
        
        default:
            ASSERT(false, "Not implemented");
    }
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(TensorElemType t, float val)
{
    return ACCCDSLImpl::_constantValToConstantImpl(t, val);
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(TensorElemType t, int32_t val)
{
    return ACCCDSLImpl::_constantValToConstantImpl(t, val);
}


std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(int64_t val)
{
    return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstInt64(val));
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(uint64_t val)
{
    return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstUInt64(val));
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(float val, bool isHalf)
{
    if (isHalf)
        return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstFloat16(val));
    else
        return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstFloat32(val));
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(double val)
{
    return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstFloat64(val));
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(int32_t val)
{
    return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstInt32(val));
}

std::shared_ptr<ACCCDSLImpl::ExpressionImpl> ACCCDSLImpl::constantValToConstantImpl(uint32_t val)
{
    return std::shared_ptr<ACCCDSLImpl::ExpressionImpl>(new ACCCDSLImpl::ConstUInt32(val));
}

std::string ACCCDSLImpl::TensorElemTypeToStr(TensorElemType t) 
{
    #define PROCESS_VAL(p) case(p): return #p; break;
    switch(t) { 
        PROCESS_VAL(None)
        PROCESS_VAL(Float16)
        PROCESS_VAL(Float32)
        PROCESS_VAL(Float64)
        PROCESS_VAL(Int8)
        PROCESS_VAL(Int16)
        PROCESS_VAL(Int32)
        PROCESS_VAL(Int64)
        PROCESS_VAL(UInt8)
        PROCESS_VAL(UInt16)
        PROCESS_VAL(UInt32)
        PROCESS_VAL(UInt64)

        default:
            ASSERT(false, "Undefined AstNodeType");
    }

    return "";
}

std::string ACCCDSLImpl::AstNodeTypeToStr(AstNodeType t) 
{
    #define PROCESS_VAL(p) case(p): return #p; break;
    switch(t) { 
        PROCESS_VAL(TensorNode)
        PROCESS_VAL(BinaryPointwiseOpNode)
        PROCESS_VAL(UnaryPointwiseOpNode)
        PROCESS_VAL(ReduceNode)
        PROCESS_VAL(AllReduceNode)
        PROCESS_VAL(BroadcastNode)
        PROCESS_VAL(AllGatherNode)
        PROCESS_VAL(ReduceTensorNode)
        PROCESS_VAL(ReduceScatterNode)
        PROCESS_VAL(StageNode)
        PROCESS_VAL(VariableNode)
        PROCESS_VAL(ConstantNode)
        PROCESS_VAL(ScatterNode)
        PROCESS_VAL(IteNode)
        PROCESS_VAL(UpdateNode)
        
        default:
            ASSERT(false, "Undefined AstNodeType " << t);
    }

    return "";
}

CollCommOperation<AllReduce_> ACCCDSL::AllReduce(AllReduceOp);
CollCommOperation<AllGather_> ACCCDSL::AllGather(AllGatherOp);
CollCommOperation<ReduceScatter_> ACCCDSL::ReduceScatter(ReduceScatterOp);
CollCommOperation<Broadcast_> ACCCDSL::Broadcast(BroadcastOp);
CollCommOperation<Scatter_> ACCCDSL::Scatter(ScatterOp);

template<typename T>
Const<T> valToConstNode(T v) {
    if (std::is_same<T, float>::value)
        return Const<T>(Float32, v);
    else if (std::is_same<T, double>::value)
        return Const<T>(Float64, v);
    else if (std::is_same<T, int>::value)
        return Const<T>(Int32, v);
    else if (std::is_same<T, uint32_t>::value)
        return Const<T>(UInt32, v);
    else if (std::is_same<T, int64_t>::value)
        return Const<T>(Int64, v);
    else if (std::is_same<T, uint64_t>::value)
        return Const<T>(UInt64, v);
    
    ASSERT(false, "Fill");
    return Const<T>(Float32, (float)v);
}

template<class T, class U, ACCCDSLImpl::BinaryOp op>
U genericConstWithClassOperatorOverload(T v, U x) {
    std::shared_ptr<ACCCDSLImpl::ExpressionImpl> constantNode;
    constantNode = valToConstNode(v).impl();
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, constantNode, 
                                                   x.impl(), x.impl()->scattered());
    return U(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

template<class T, class U, ACCCDSLImpl::BinaryOp op>
T genericClassWithConstOperatorOverload(T v, U x) {
    std::shared_ptr<ACCCDSLImpl::ExpressionImpl> constantNode;
    constantNode = valToConstNode(x).impl();
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, v.impl(), constantNode, v.impl()->scattered());
    return T(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

template<typename T, ACCCDSLImpl::BinaryOp op>
T genericClassOpVariableOverload(Expression& v, Expression& x) {
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, v.impl(), x.impl(), false);
    return T(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

template<typename T, ACCCDSLImpl::UnaryOp op>
T genericClassUnaryOp(Expression& x) {
    auto node = new ACCCDSLImpl::UnaryPointwiseOp(op, x.impl());
    return T(std::shared_ptr<ACCCDSLImpl::UnaryPointwiseOp>(node));
}

ContinuousExpression ACCCDSL::operator-(float v, ContinuousExpression x) {
    return genericConstWithClassOperatorOverload<float, ContinuousExpression, ACCCDSLImpl::BinaryOp::Subtract>(v, x);
}

ContinuousExpression ACCCDSL::operator+(float v, ContinuousExpression x) {
    return genericConstWithClassOperatorOverload<float, ContinuousExpression, ACCCDSLImpl::BinaryOp::Add>(v, x);
}

ContinuousExpression ACCCDSL::operator*(float v, ContinuousExpression x) {
    return genericConstWithClassOperatorOverload<float, ContinuousExpression, ACCCDSLImpl::BinaryOp::Multiply>(v, x);
}

ContinuousExpression ACCCDSL::operator/(float v, ContinuousExpression x) {
    return genericConstWithClassOperatorOverload<float, ContinuousExpression, ACCCDSLImpl::BinaryOp::Divide>(v, x);
}

/*Operators with left operand as ContinuousExpression and right as constant*/
ContinuousExpression ACCCDSL::operator-(ContinuousExpression x, float v) {
    return genericClassWithConstOperatorOverload<ContinuousExpression, float, ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}
ContinuousExpression ACCCDSL::operator+(ContinuousExpression x, float v) {
    return genericClassWithConstOperatorOverload<ContinuousExpression, float, ACCCDSLImpl::BinaryOp::Add>(x, v);
}
ContinuousExpression ACCCDSL::operator*(ContinuousExpression x, float v) {
    return genericClassWithConstOperatorOverload<ContinuousExpression, float, ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}
ContinuousExpression ACCCDSL::operator/(ContinuousExpression x, float v) {
    return genericClassWithConstOperatorOverload<ContinuousExpression, float, ACCCDSLImpl::BinaryOp::Divide>(x, v);
}

ContinuousExpression ACCCDSL::operator>(ContinuousExpression x, float v) {
    return genericClassWithConstOperatorOverload<ContinuousExpression, float, ACCCDSLImpl::BinaryOp::Greater>(x, v);
}
SingleDimExpression ACCCDSL::operator>(SingleDimExpression x, float v) {
    return genericClassWithConstOperatorOverload<SingleDimExpression, float, ACCCDSLImpl::BinaryOp::Greater>(x, v);
}

ContinuousExpression ACCCDSL::operator-(ContinuousExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}
ContinuousExpression ACCCDSL::operator+(ContinuousExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Add>(x, v);
}
ContinuousExpression ACCCDSL::operator*(ContinuousExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}
ContinuousExpression ACCCDSL::operator/(ContinuousExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Divide>(x, v);
}

ContinuousExpression ACCCDSL::operator-(SingleDimExpression v, ContinuousExpression x) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}
ContinuousExpression ACCCDSL::operator+(SingleDimExpression v, ContinuousExpression x) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Add>(x, v);
}
ContinuousExpression ACCCDSL::operator*(SingleDimExpression v, ContinuousExpression x) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}
ContinuousExpression ACCCDSL::operator/(SingleDimExpression v, ContinuousExpression x) {
    return genericClassOpVariableOverload<ContinuousExpression, ACCCDSLImpl::BinaryOp::Divide>(x, v);
}

ScatteredExpression ACCCDSL::operator-(float v, ScatteredExpression x) {
    return genericConstWithClassOperatorOverload<float, ScatteredExpression, ACCCDSLImpl::BinaryOp::Subtract>(v, x);
}

ScatteredExpression ACCCDSL::operator+(float v, ScatteredExpression x) {
    return genericConstWithClassOperatorOverload<float, ScatteredExpression, ACCCDSLImpl::BinaryOp::Add>(v, x);
}

ScatteredExpression ACCCDSL::operator*(float v, ScatteredExpression x) {
    return genericConstWithClassOperatorOverload<float, ScatteredExpression, ACCCDSLImpl::BinaryOp::Multiply>(v, x);
}

ScatteredExpression ACCCDSL::operator/(float v, ScatteredExpression x) {
    return genericConstWithClassOperatorOverload<float, ScatteredExpression, ACCCDSLImpl::BinaryOp::Divide>(v, x);
}

/*Operators with left operand as ContinuousExpression and right as constant*/
ScatteredExpression ACCCDSL::operator-(ScatteredExpression x, float v) {
    return genericClassWithConstOperatorOverload<ScatteredExpression, float, ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}

ScatteredExpression ACCCDSL::operator+(ScatteredExpression x, float v) {
    return genericClassWithConstOperatorOverload<ScatteredExpression, float, ACCCDSLImpl::BinaryOp::Add>(x, v);
}

ScatteredExpression ACCCDSL::operator*(ScatteredExpression x, float v) {
    return genericClassWithConstOperatorOverload<ScatteredExpression, float, ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}

ScatteredExpression ACCCDSL::operator/(ScatteredExpression x, float v) {
    return genericClassWithConstOperatorOverload<ScatteredExpression, float, ACCCDSLImpl::BinaryOp::Divide>(x, v);
}

/*Variable 'op' Scattered is again a scattered expression */
template<ACCCDSLImpl::BinaryOp op>
ScatteredExpression genericClassOpVariableOverload(ScatteredExpression v, SingleDimExpression x) {
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, v.impl(), x.impl(), true);
    return ScatteredExpression(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

template<ACCCDSLImpl::BinaryOp op>
ScatteredExpression genericVariableOpClassOverload(SingleDimExpression x, ScatteredExpression v) {
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, x.impl(), v.impl(), true);
    return ScatteredExpression(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

template<ACCCDSLImpl::BinaryOp op>
SingleDimExpression genericSingleDimExpressionOp(SingleDimExpression x, SingleDimExpression v) {
    auto node = new ACCCDSLImpl::BinaryPointwiseOp(op, x.impl(), v.impl(), false);
    return SingleDimExpression(std::shared_ptr<ACCCDSLImpl::BinaryPointwiseOp>(node));
}

ScatteredExpression ACCCDSL::operator-(ScatteredExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}


ScatteredExpression ACCCDSL::operator*(ScatteredExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}


ScatteredExpression ACCCDSL::operator/(ScatteredExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ACCCDSLImpl::BinaryOp::Divide>(x, v);
}


ScatteredExpression ACCCDSL::operator+(ScatteredExpression x, SingleDimExpression v) {
    return genericClassOpVariableOverload<ACCCDSLImpl::BinaryOp::Add>(x, v);
}

ScatteredExpression ACCCDSL::operator-(SingleDimExpression x, ScatteredExpression v) {
    return genericVariableOpClassOverload<ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}

ScatteredExpression ACCCDSL::operator+(SingleDimExpression x, ScatteredExpression v) {
    return genericVariableOpClassOverload<ACCCDSLImpl::BinaryOp::Add>(x, v);
}

ScatteredExpression ACCCDSL::operator/(SingleDimExpression x, ScatteredExpression v) {
    return genericVariableOpClassOverload<ACCCDSLImpl::BinaryOp::Divide>(x, v);
}

ScatteredExpression ACCCDSL::operator*(SingleDimExpression x, ScatteredExpression v) {
    return genericVariableOpClassOverload<ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}

SingleDimExpression ACCCDSL::operator-(float x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Subtract>(Const<float>(Float32, x), v);
}

SingleDimExpression ACCCDSL::operator*(float x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Multiply>(Const<float>(Float32, x), v);
}
SingleDimExpression ACCCDSL::operator/(float x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Divide>(Const<float>(Float32, x), v);
}
SingleDimExpression ACCCDSL::operator+(float x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Add>(Const<float>(Float32, x), v);
}
SingleDimExpression ACCCDSL::operator-(SingleDimExpression x, float v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Subtract>(x, Const<float>(Float32, v));
}
SingleDimExpression ACCCDSL::operator*(SingleDimExpression x, float v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Multiply>(x, Const<float>(Float32, v));
}
SingleDimExpression ACCCDSL::operator/(SingleDimExpression x, float v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Divide>(x, Const<float>(Float32, v));
}
SingleDimExpression ACCCDSL::operator+(SingleDimExpression x, float v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Add>(x, Const<float>(Float32, v));
}

SingleDimExpression ACCCDSL::operator-(SingleDimExpression x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Subtract>(x, v);
}
SingleDimExpression ACCCDSL::operator*(SingleDimExpression x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Multiply>(x, v);
}
SingleDimExpression ACCCDSL::operator/(SingleDimExpression x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Divide>(x, v);
}
SingleDimExpression ACCCDSL::operator+(SingleDimExpression x, SingleDimExpression v) {
    return genericSingleDimExpressionOp<ACCCDSLImpl::BinaryOp::Add>(x, v);
}

ContinuousExpression ACCCDSL::Sqrt(ContinuousExpression x) {
    return genericClassUnaryOp<ContinuousExpression, ACCCDSLImpl::UnaryOp::SqrtOp>(x);
}

ContinuousExpression ACCCDSL::Ite(ContinuousExpression cond, ContinuousExpression ifTrue, ContinuousExpression ifFalse) {
    return ContinuousExpression(std::make_shared<ACCCDSLImpl::IteImpl>(cond.impl(), ifTrue.impl(), ifFalse.impl(), false));
}

ContinuousExpression ACCCDSL::Ite(ContinuousExpression cond, ContinuousExpression ifTrue, float ifFalse) {
    return ContinuousExpression(std::make_shared<ACCCDSLImpl::IteImpl>(cond.impl(), ifTrue.impl(), valToConstNode(ifFalse).impl(), false));
}

ContinuousExpression ACCCDSL::Ite(SingleDimExpression cond, ContinuousExpression ifTrue, float ifFalse) {
    return ContinuousExpression(std::make_shared<ACCCDSLImpl::IteImpl>(cond.impl(), ifTrue.impl(), valToConstNode(ifFalse).impl(), false));
}

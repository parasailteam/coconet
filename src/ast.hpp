#ifndef __AST_HPP__
#define __AST_HPP__

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <queue>

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step)
{
  if (step == IntType(0))
  {
    throw std::invalid_argument("step for range must be non-zero");
  }

  std::vector<IntType> result;
  IntType i = start;
  while ((step > 0) ? (i < stop) : (i > stop))
  {
    result.push_back(i);
    i += step;
  }

  return result;
}

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

namespace ACCCDSL
{
enum TensorLayout {
    Replicated,
    Sliced,
    Sliced_2,
    Local
};

enum AstNodeType {
    TensorNode,
    BinaryPointwiseOpNode,
    UnaryPointwiseOpNode,
    ReduceNode,
    AllReduceNode,
    BroadcastNode,
    AllGatherNode,
    ReduceTensorNode,
    SendNode,
    NormNode,
    ReduceScatterNode,
    FusedAllReduceNone,
    DropoutNode,
    StageNode,
    UpdateNode,
    FusedNode,
    VariableNode,
    ProcessGroupNode,
    ProcessGroupIDNode,
    ConstantNode,
    CastNode,
    ScatterNode,
    MatMulNode,
    IteNode,
};

enum TensorElemType {
    None,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    //MPI and NCCL Types
    MPIComm,
    NCCLComm
};

enum ReduceOperation {
    ReduceOperationNone,
    Summation,
    Difference,
    Multiplication,
    Division,
    Maximum,
    Minimum
};
} // namespace ACCLDSL

namespace ACCCDSLImpl{

using namespace ACCCDSL;

class AstVisitor;

class AstNodeImpl;
class AllReduceImpl;
class ReduceImpl;
class AllGatherImpl;
class BroadcastImpl;
class ReduceScatterImpl;
class FusedAllReduceImpl;
class BinaryPointwiseOp;
class MatMulImpl;
class CommCollPrimitiveImpl;
class ExpressionImpl;
class PowerImpl;
class ReduceTensorImpl;
class SendImpl;
class StageImpl;
class UpdateImpl;
class NormImpl;
class FusedNode;
class TensorImpl;
class UnaryPointwiseOp;
class VariableImpl;
class ProcessGroupImpl;
class ProcessGroupIDImpl;
class CastImpl;
template<class T>
class ConstantImpl;
class ConstUInt64;
class ConstInt64;
class ConstFloat16;
class ConstFloat32;
class ConstFloat64;
class ConstUInt32;
class ConstInt32;
class ScatterImpl;
class IteImpl;
class DropoutImpl;

extern std::shared_ptr<ProcessGroupImpl> WORLDGroupimpl;
extern std::shared_ptr<ProcessGroupIDImpl> WORLDimpl;
std::string AstNodeTypeToStr(AstNodeType t);
std::string TensorElemTypeToStr(TensorElemType t);

class AstNodeImpl {
protected:
    AstNodeType type_;
    AstNodeImpl(AstNodeType type) : type_(type) {}
    AstNodeImpl(AstNodeType type, std::shared_ptr<AstNodeImpl> child) : type_(type) {
        children_.push_back(child);
    }
    AstNodeImpl(AstNodeType type, std::shared_ptr<AstNodeImpl> child1, std::shared_ptr<AstNodeImpl> child2) : type_(type) {
        children_.push_back(child1);
        children_.push_back(child2);
    }
    AstNodeImpl(AstNodeType type, std::initializer_list<std::shared_ptr<AstNodeImpl>> children) : type_(type), children_(children) {}
    std::vector<std::shared_ptr<AstNodeImpl>> children_;
    friend AstVisitor;
    static int nameCounter;
public:
    bool isCommCollective() {return type_ == AllReduceNode || type_ == AllGatherNode || type_ == ReduceNode ||
                             type_ == ReduceScatterNode || type_ == BroadcastNode;}
    bool isConstant() {return type_ == ConstantNode;}
    virtual void accept(AstVisitor& v) = 0;
    AstNodeType type() {return type_;}
    #define asChildSharedPtr(x) \
        static std::shared_ptr<x> as##x(std::shared_ptr<AstNodeImpl> y){return std::dynamic_pointer_cast<x>(y);} 
    
    // #define asChildRawPtr(x) \
    //     static x* as##x(AstNodeImpl* y){x* ptr; ASSERT(, "Bad casting"); ptr = (x*)(y); \
    //         return ptr;} 

    bool hasChild(std::shared_ptr<AstNodeImpl> oldChild)
    {return std::find(children_.begin(), children_.end(), oldChild) != children_.end();}
    std::vector<std::shared_ptr<AstNodeImpl>>::iterator findChild(std::shared_ptr<AstNodeImpl> oldChild) 
    {return std::find(children_.begin(), children_.end(), oldChild);}
    void replaceChildren(std::vector<std::shared_ptr<AstNodeImpl>>::iterator& iter, std::shared_ptr<AstNodeImpl> newChild)
    {children_[iter - children_.begin()] = newChild;}
    const std::vector<std::shared_ptr<AstNodeImpl>>& children() {return children_;}
    #define asChild(x) asChildSharedPtr(x)

    asChild(AllReduceImpl)
    asChild(ReduceImpl);
    asChild(AllGatherImpl);
    asChild(BroadcastImpl);
    asChild(ReduceScatterImpl);
    asChild(BinaryPointwiseOp);
    asChild(MatMulImpl);
    asChild(CommCollPrimitiveImpl);
    asChild(ExpressionImpl);
    asChild(PowerImpl);
    asChild(ReduceTensorImpl);
    asChild(StageImpl);
    asChild(UpdateImpl);
    asChild(TensorImpl);
    asChild(ScatterImpl);
    asChild(UnaryPointwiseOp);
    asChild(VariableImpl);
    asChild(ProcessGroupImpl);
    asChild(ProcessGroupIDImpl);
    asChild(CastImpl);
    asChild(ConstUInt64);
    asChild(ConstInt64);
    asChild(ConstFloat16);
    asChild(ConstFloat32);
    asChild(ConstFloat64);
    asChild(ConstUInt32);
    asChild(ConstInt32);
    asChild(IteImpl);
    asChild(NormImpl);
    asChild(DropoutImpl);
    asChild(SendImpl);
};

class AstVisitor {
protected:
    AstVisitor() {}
public:
    virtual void visit(TensorImpl& node) = 0;
    virtual void visit(AllReduceImpl& node) = 0;
    virtual void visit(ReduceImpl& node) = 0;
    virtual void visit(BroadcastImpl& node) = 0;
    virtual void visit(AllGatherImpl& node) = 0;
    virtual void visit(ReduceScatterImpl& node) = 0;
    virtual void visit(BinaryPointwiseOp& node) = 0;
    virtual void visit(MatMulImpl& node) = 0;
    virtual void visit(UnaryPointwiseOp& node) = 0;
    virtual void visit(PowerImpl& node) = 0;
    virtual void visit(ReduceTensorImpl& node) = 0;
    virtual void visit(SendImpl& node) = 0;
    virtual void visit(NormImpl& node) = 0;
    virtual void visit(DropoutImpl& node) = 0;
    virtual void visit(StageImpl& node) = 0;
    virtual void visit(UpdateImpl& node) = 0;
    virtual void visit(VariableImpl& node) = 0;
    virtual void visit(ProcessGroupImpl& node) = 0;
    virtual void visit(ProcessGroupIDImpl& node) = 0;
    virtual void visit(CastImpl& node) = 0;
    virtual void visit(ConstUInt64& node) = 0;
    virtual void visit(ConstInt64& node) = 0;
    virtual void visit(ConstUInt32& node) = 0;
    virtual void visit(ConstInt32& node) = 0;
    virtual void visit(ConstFloat16& node) = 0;
    virtual void visit(ConstFloat32& node) = 0;
    virtual void visit(ConstFloat64& node) = 0;
    virtual void visit(ScatterImpl& node) = 0;
    virtual void visit(IteImpl& node) = 0;

    void visitChildren(AstNodeImpl& node) {
        for (auto child : node.children_) {
            child->accept(*this);
        }
    }
};

class ExpressionImpl : public AstNodeImpl {
protected:
    //Size of each dimension
    std::vector<std::shared_ptr<ExpressionImpl>> dimSizes_;
    //Layout
    TensorLayout layout_;
    //Name of the expression. Currently only a tensor and variable 
    //has names.
    std::string name_;
    //Type of tensor element.
    TensorElemType elemType_;
    //Process Group
    std::shared_ptr<ProcessGroupIDImpl> group_;

public:
    ExpressionImpl(AstNodeType type, std::shared_ptr<ProcessGroupIDImpl> group) : group_(group), AstNodeImpl(type) {}
    ExpressionImpl(AstNodeType type, std::shared_ptr<ExpressionImpl> child, std::shared_ptr<ProcessGroupIDImpl> group) : 
        group_(group), AstNodeImpl(type, child)
    {
    }

    ExpressionImpl(AstNodeType type, std::shared_ptr<AstNodeImpl> child1, std::shared_ptr<AstNodeImpl> child2,
                  std::shared_ptr<ProcessGroupIDImpl> group) : 
        group_(group), AstNodeImpl(type, child1, child2)
    {
    }

    ExpressionImpl(AstNodeType type, std::initializer_list<std::shared_ptr<AstNodeImpl>> children,
                   std::shared_ptr<ProcessGroupIDImpl> group) : 
        group_(group), AstNodeImpl(type, children)
    {
    }
    
    bool scattered() {return layout_ == Sliced;}
    virtual void setupAndCheckDimensions() = 0;
    std::string name() {return name_;}
    virtual void accept(AstVisitor& v) = 0;
    virtual TensorLayout layout () {return layout_;}
    virtual size_t dims() {return dimSizes_.size();}
    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return dimSizes_[dim];}
    virtual TensorElemType elemType() {return elemType_;}
    virtual const std::vector<std::shared_ptr<ExpressionImpl>>& dimSizes() {return dimSizes_;}
    std::shared_ptr<ProcessGroupIDImpl> group() {return group_;}

    bool isPointwise();

    template<typename T>
    std::set<std::shared_ptr<T>> childrenOfType()
    {
        std::set<std::shared_ptr<T>> used;
        std::queue<std::shared_ptr<AstNodeImpl>> exprQueue;
        std::unordered_set<std::shared_ptr<AstNodeImpl>> visitedExprs;

        for (auto child : children())
            exprQueue.push(child);

        while (!exprQueue.empty()) {
            auto expr = exprQueue.front();
            exprQueue.pop();
            
            if (visitedExprs.count(expr) > 0)
                continue;

            if (std::dynamic_pointer_cast<T>(expr) != nullptr) {
                used.insert(std::dynamic_pointer_cast<T>(expr));
            } else {
                for (auto child : expr->children())
                    exprQueue.push(child);
            }

            visitedExprs.insert(expr);
        }

        return used;
    }

    std::set<std::shared_ptr<ExpressionImpl>> usedExprs()
    {
        std::set<std::shared_ptr<ExpressionImpl>> used;
        std::queue<std::shared_ptr<AstNodeImpl>> exprQueue;
        std::unordered_set<std::shared_ptr<AstNodeImpl>> visitedExprs;

        for (auto child : children())
            exprQueue.push(child);

        while (!exprQueue.empty()) {
            auto expr = exprQueue.front();
            exprQueue.pop();
            
            if (visitedExprs.count(expr) > 0)
                continue;

            if (expr->type() == StageNode || expr->type() == TensorNode || expr->type() == VariableNode) {
                used.insert(AstNodeImpl::asExpressionImpl(expr));
            } else {
                for (auto child : expr->children())
                    exprQueue.push(child);
            }

            visitedExprs.insert(expr);
        }

        return used;
    }

    virtual ~ExpressionImpl() {}
};


template<class T>
class ConstantImpl : public ExpressionImpl {
private:
    const T val_;
public:
    //A constant is never scattered but is available on all GPUs
    ConstantImpl(T val) : val_(val), ExpressionImpl(AstNodeType::ConstantNode, WORLDimpl) {setupAndCheckDimensions();}
    T val() {return val_;}

    virtual std::string name() {
        return std::to_string(val_);
    }
    virtual size_t dims() {return 1;}
    virtual void setupAndCheckDimensions() {
        dimSizes_.clear();
        // dimSizes_.push_back(std::shared_ptr<ConstantImpl<int>>(new std::shared_ptr<ConstantImpl<int>>(1)));
    }
};

class ConstInt64 : public ConstantImpl<long> {
public:
    ConstInt64(long val) : ConstantImpl<long> (val) {
        elemType_ = Int64;
    }
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class ConstUInt64 : public ConstantImpl<uint64_t> {
public:
    ConstUInt64(uint64_t val) : ConstantImpl<uint64_t> (val) {elemType_ = UInt64;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class ConstInt32 : public ConstantImpl<int> {
public:
    ConstInt32(int val) : ConstantImpl<int> (val) {elemType_ = Int32;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class ConstUInt32 : public ConstantImpl<uint32_t> {
public:
    ConstUInt32(uint32_t val) : ConstantImpl<uint32_t> (val) {elemType_ = UInt32;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class ConstFloat16 : public ConstantImpl<float> {
public:
    ConstFloat16(float val) : ConstantImpl<float> (val) {elemType_ = Float16;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};


class ConstFloat32 : public ConstantImpl<float> {
public:
    ConstFloat32(float val) : ConstantImpl<float> (val) {elemType_ = Float32;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class ConstFloat64 : public ConstantImpl<double> {
public:
    ConstFloat64(double val) : ConstantImpl<double> (val) {elemType_ = Float64;}
     virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

class TensorImpl : public ExpressionImpl {
public:
    TensorImpl(TensorElemType elemType, std::shared_ptr<ExpressionImpl> n1, std::shared_ptr<ExpressionImpl> n2, TensorLayout layout, std::string name, std::shared_ptr<ProcessGroupIDImpl> group) : 
        ExpressionImpl(AstNodeType::TensorNode, group)
    {
        dimSizes_ = {n1, n2};
        layout_ = layout;
        elemType_ = elemType;
        name_ = name;
    }

    TensorImpl(TensorElemType elemType, std::shared_ptr<ExpressionImpl> n, TensorLayout layout, std::string name,
               std::shared_ptr<ProcessGroupIDImpl> group) : 
        ExpressionImpl(AstNodeType::TensorNode, group)
    {
        dimSizes_  = {n};
        layout_ = layout;
        elemType_ = elemType;
        name_ = name;
    }

    TensorImpl(TensorElemType elemType, std::vector<std::shared_ptr<ExpressionImpl>> dimSizes, TensorLayout layout, 
               std::string name, std::shared_ptr<ProcessGroupIDImpl> group) : 
        ExpressionImpl(AstNodeType::TensorNode, group)
    {
        elemType_ = elemType; 
        dimSizes_ = dimSizes; 
        layout_ = layout;
        name_ = name;
    }

    std::string name() {return name_;}

    virtual void setupAndCheckDimensions() {/*Nothing to setup because Tensor do not have any children*/}
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    void setName(std::string newName) {name_ = newName;}
    TensorImpl copyWithNewName(std::string newName) 
    {
        TensorImpl newS = *this;
        newS.setName(newName);
        return newS;
    }
    void addDim(std::shared_ptr<ExpressionImpl> d) {dimSizes_.push_back(d);}
};

class VariableImpl : public TensorImpl {
public:
    //A variable is never scattered but is present on all gpus
    VariableImpl(TensorElemType t, std::string name) : 
        TensorImpl(t, std::shared_ptr<ConstInt32>(new ConstInt32(1)), Replicated, name, WORLDimpl)
    {
        type_ = AstNodeType::VariableNode;
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};

enum ProcessGroupIDType {
    PreviousProcessGroupID,
    CurrentProcessGroupID,
    NextProcessGroupID,
};

class ProcessGroupImpl : public ExpressionImpl {
private:
    std::shared_ptr<VariableImpl> sizeVar_;
    std::shared_ptr<VariableImpl> rankVar_;
    std::shared_ptr<VariableImpl> mpiCommVar_;
    std::shared_ptr<VariableImpl> ncclCommVar_;
    static int nameCounter;
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> id_;
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> nextGroup_;
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> previousGroup_;

public:
    //A variable is never scattered but is present on all gpus
    ProcessGroupImpl(std::shared_ptr<ProcessGroupImpl> parent, std::shared_ptr<ExpressionImpl> splitSize) : 
    ExpressionImpl(AstNodeType::ProcessGroupNode, parent, splitSize, WORLDimpl)
    {
        nameCounter++;
        sizeVar_ = std::shared_ptr<VariableImpl>(new VariableImpl(Int32, "size_"+name()));
        rankVar_ = std::shared_ptr<VariableImpl>(new VariableImpl(Int32, "rank_"+name()));
        mpiCommVar_ = std::shared_ptr<VariableImpl>(new VariableImpl(MPIComm, (parent) ? "mpiComm_"+name() : "MPI_COMM_WORLD"));
        ncclCommVar_ = std::shared_ptr<VariableImpl>(new VariableImpl(NCCLComm, "ncclComm_"+name()));
        children_.push_back(sizeVar_);
        children_.push_back(rankVar_);
        children_.push_back(mpiCommVar_);
        children_.push_back(ncclCommVar_);

        id_ = nullptr;
        nextGroup_ = nullptr;
        previousGroup_ = nullptr;
        setupAndCheckDimensions();
    }

    std::shared_ptr<VariableImpl> sizeVar() {return sizeVar_;}
    std::shared_ptr<VariableImpl> rankVar() {return rankVar_;}
    std::shared_ptr<VariableImpl> mpiCommVar() {return mpiCommVar_;}
    std::shared_ptr<VariableImpl> ncclCommVar() {return ncclCommVar_;}

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ProcessGroupImpl> parent() {return std::dynamic_pointer_cast<ProcessGroupImpl>(children_[0]);}
    std::shared_ptr<ExpressionImpl> splitSize() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[1]);}

    virtual void setupAndCheckDimensions() 
    {
        dimSizes_.clear();
        dimSizes_.push_back(std::shared_ptr<ConstInt32>(new ConstInt32(1)));
        elemType_ = Int32;
        layout_ = Local;
    }

    std::string name() {return (parent() == nullptr) ? splitSize()->name() : "group_"+std::to_string(nameCounter);}
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> id(std::shared_ptr<ACCCDSLImpl::ProcessGroupImpl> ptr);
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> nextGroup(std::shared_ptr<ACCCDSLImpl::ProcessGroupImpl> ptr);
    std::shared_ptr<ACCCDSLImpl::ProcessGroupIDImpl> prevGroup(std::shared_ptr<ACCCDSLImpl::ProcessGroupImpl> ptr);
};

class ProcessGroupIDImpl : public ExpressionImpl {
    private:
        ProcessGroupIDType type_;

    public:
        ProcessGroupIDImpl(std::shared_ptr<ProcessGroupImpl> group, ProcessGroupIDType type): type_(type), ExpressionImpl(AstNodeType::ProcessGroupIDNode, asExpressionImpl(group), WORLDimpl) {}
        
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ProcessGroupImpl> group() {return AstNodeImpl::asProcessGroupImpl(children_[0]);}
    ProcessGroupIDType idType() {return type_;}

    virtual void setupAndCheckDimensions() 
    {
        dimSizes_.clear();
        dimSizes_.push_back(std::shared_ptr<ConstInt32>(new ConstInt32(1)));
        elemType_ = Int32;
        layout_ = Local;
    }
};

class StageImpl : public ExpressionImpl {
public:
    StageImpl(std::shared_ptr<ExpressionImpl> definition)  : 
        ExpressionImpl(AstNodeType::StageNode, definition, definition->group()) 
    {
        name_ = "S" + std::to_string(nameCounter++);
        setupAndCheckDimensions();
    }
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    std::shared_ptr<ExpressionImpl> definition() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    std::string name() {return name_;}

    virtual void setupAndCheckDimensions() {
        dimSizes_.clear();
        for(size_t s = 0; s < definition()->dims(); s++) {
            dimSizes_.push_back(definition()->size(s));
        }
        
        layout_ = definition()->layout();
        elemType_ = definition()->elemType();
    }

    void setName(std::string name) {name_ = name;}
    StageImpl copyWithNewName(std::string newName) 
    {
        StageImpl newS = *this;
        newS.setName(newName);
        return newS;
    }

    std::set<std::shared_ptr<StageImpl>> dependsOnStages()
    {
        std::set<std::shared_ptr<StageImpl>> dependent;
        std::queue<std::shared_ptr<AstNodeImpl>> exprQueue;
        std::unordered_set<std::shared_ptr<AstNodeImpl>> visitedStages;

        exprQueue.push(definition());

        while (!exprQueue.empty()) {
            auto expr = exprQueue.front();
            exprQueue.pop();

            auto used = AstNodeImpl::asExpressionImpl(expr)->usedExprs();
            for (auto u : used) {
                if (u->type() == StageNode) {
                    dependent.insert(AstNodeImpl::asStageImpl(u));
                    exprQueue.push(u);
                }
            }                
        }

        return dependent;
    }
};

template<class T> 
std::shared_ptr<ExpressionImpl> _constantValToConstantImpl(TensorElemType t, T val);

std::shared_ptr<ExpressionImpl> constantValToConstantImpl(TensorElemType t, float val);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(TensorElemType t, int32_t val);

std::shared_ptr<ExpressionImpl> constantValToConstantImpl(int64_t val);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(uint64_t val);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(float val, bool isHalf);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(double val);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(int32_t val);
std::shared_ptr<ExpressionImpl> constantValToConstantImpl(uint32_t val);

enum BinaryOp {
    Add,
    Multiply,
    Divide,
    Subtract,
    Greater,
};

class BinaryPointwiseOp : public ExpressionImpl {
protected:
    BinaryOp op_;
public:
    BinaryPointwiseOp(BinaryOp op, std::shared_ptr<ExpressionImpl> t1, std::shared_ptr<ExpressionImpl> t2) :
        ExpressionImpl(AstNodeType::BinaryPointwiseOpNode, t1, t2, nullptr), op_(op) 
    {
        setupAndCheckDimensions();    
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> operand(int i) {
        ASSERT (i<2, "Binary operation do not have more than 2 children."); 
        return std::dynamic_pointer_cast<ExpressionImpl>(children_[i]);
    }

    BinaryOp op() {return op_;}

    static std::string operatorToStr(BinaryOp op) {
        switch (op) {
            case Add:
                return "+";
            case Subtract:
                return "-";
            case Divide:
                return "/";
            case Multiply:
                return "*";
            case Greater:
                return ">";
            default:
                ASSERT(false, "Invalid BinaryOp '"<< op <<"'");
                return "";
        }
    }

    static std::string operatorToHalfFunc(BinaryOp op) {
        switch (op) {
            case Add:
                return "__hadd";
            case Subtract:
                return "__hsub";
            case Divide:
                return "__hdiv";
            case Multiply:
                return "__hmul";
            case Greater:
                return "__hgt";

            default:
                ASSERT(false, "Invalid BinaryOp '"<< op <<"'");
                return "";
        }
    }

    static std::string operatorToHalf2Func(BinaryOp op) {
        switch (op) {
            case Add:
                return "__hadd2";
            case Subtract:
                return "__hsub2";
            case Divide:
                return "__h2div";
            case Multiply:
                return "__hmul2";
            case Greater:
                return "__hgt2";
                
            default:
                ASSERT(false, "Invalid BinaryOp '"<< op <<"'");
                return "";
        }
    }

    virtual void setupAndCheckDimensions();
};

// class Sqrt : public AstNode {
// public:
//     Sqrt(std::shared_ptr<TensorImpl> t) : AstNode(AstNodeType::SqrtNode, t) {}
// };

enum UnaryOp {
    PowerOp, // TODO: Power isn't a unary operation!!
    SqrtOp
};

class UnaryPointwiseOp : public ExpressionImpl {
protected:
    UnaryOp op_;
public:
    UnaryPointwiseOp(UnaryOp op, std::shared_ptr<ExpressionImpl> t) :
        ExpressionImpl(AstNodeType::UnaryPointwiseOpNode, t, t->group()), op_(op)
    {
        setupAndCheckDimensions();
    }

    static std::string operatorToStr(UnaryOp op) {
        switch (op) {
            case SqrtOp:
                return "sqrt";

            default:
                ASSERT(false, "Invalid UnaryOp '"<< op <<"'");
                return "";
        }
    }

    static std::string operatorToHalfFunc(UnaryOp op) {
        switch (op) {
            case SqrtOp:
                return "hsqrt";

            default:
                ASSERT(false, "Invalid UnaryOp '"<< op <<"'");
                return "";
        }
    }

    static std::string operatorToHalf2Func(UnaryOp op) {
        switch (op) {
            case SqrtOp:
                return "h2sqrt";
                
            default:
                ASSERT(false, "Invalid UnaryOp '"<< op <<"'");
                return "";
        }
    }
    
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> operand() {
        return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);
    }

    UnaryOp op() { return op_; }

    virtual size_t dims() {return operand()->dims();};
    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return operand()->size(dim);}
    virtual TensorLayout layout() {return operand()->layout();}
    
    virtual void setupAndCheckDimensions() 
    {
        dimSizes_.clear();
        elemType_ = operand()->elemType();
    }
};


class PowerImpl : public UnaryPointwiseOp {
protected:
    float n_;
public:
    PowerImpl(std::shared_ptr<ExpressionImpl> t, float n) : 
        UnaryPointwiseOp(UnaryOp::PowerOp, t), n_(n) {}

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    float n() {return n_;}
};

class ScatterImpl : public ExpressionImpl {
private:
public:
    ScatterImpl(std::shared_ptr<TensorImpl> t) : 
        ExpressionImpl(AstNodeType::ScatterNode, t, t->group()) 
    {
        setupAndCheckDimensions();
    }
    ScatterImpl(std::shared_ptr<StageImpl> s) : 
        ExpressionImpl(AstNodeType::ScatterNode, std::static_pointer_cast<ExpressionImpl>(s), s->group())
    {
        setupAndCheckDimensions();
    }
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return arg()->dims();
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);};
    virtual const std::vector<std::shared_ptr<ExpressionImpl>>& dimSizes() {return arg()->dimSizes();} 
    virtual void setupAndCheckDimensions() 
    {
        //std::cout << typeid(*arg().get()).name() << std::endl;
        ASSERT(!arg()->scattered(), "Scatter does not take a scattered expression as input");
        dimSizes_.clear();
        elemType_ = arg()->elemType();
        layout_ = Sliced;
    }
};

class IteImpl : public ExpressionImpl {
private:
public:
    IteImpl(std::shared_ptr<ExpressionImpl> cond, std::shared_ptr<ExpressionImpl> ifTrue, std::shared_ptr<ExpressionImpl> ifFalse) :
        ExpressionImpl(AstNodeType::IteNode, {cond, ifTrue, ifFalse}, nullptr)
    {
        setupAndCheckDimensions();
    }
    // virtual std::shared_ptr<AstNodeImpl> clone()
    // {
    //     Ite* ite;
    //     auto argClone = this->arg()->clone();
    //     if (argClone->type() == TensorNode) {
    //         b = new ScatterImpl(AstNodeImpl::asTensorImpl(argClone));
    //     } else {
    //         b = new ScatterImpl(AstNodeImpl::asStageImpl(argClone));
    //     }
    //     return std::shared_ptr<ScatterImpl>(b);
    // }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    
    std::shared_ptr<ExpressionImpl> cond() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    std::shared_ptr<ExpressionImpl> ifTrue() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[1]);}
    std::shared_ptr<ExpressionImpl> ifFalse() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[2]);}

    virtual void setupAndCheckDimensions() 
    {
        /*Following are the dimension and gpu checking rules
         *  ExpressionType ->       |    Constant           |   Variable          | Continuous Expression |   Scattered Expression
         *    |                     |                       |                     |                       |
         *   \ /                    |                       |                     |                       |
         *   Constant               |    Always legal       |  Dims and Layout      | Dims and Layout of      |  Dims and Layout of
         *                          |                       |  of Variable        | Expression            |     Expression
         * -------------------------------------------------------------------------------------------------------------
         *   Variable               |    Dimension and gpu  |  Dims and Layout      | Dims of Expression    |   Dims of Expression
         *                          |     of Variable       |  should match       | Layout should match     |   Layout should match
         * ----------------------------------------------------------------------------------------------------------------
         *   Continuous Expression  |    Dims and gpu of    |  Dims of Expression | Dims and Layout of      |   Illegal
         *                          |     Expression        |  Layout should match  | Expressions           |
         *                          |                       |                     | should match          |
         * ----------------------------------------------------------------------------------------------------------------
         *   Scattered Expression   |    Dims and GPU of    |  Dims of Expression |    Illegal            |  Dims and Layout should
         *                          |    Expression         |  Layout should match  |                       |   match
         * 
         */ 
        
        //TODO: Set Group
        ASSERT(false, "Set group");
        dimSizes_.clear();
        
        if (ifTrue()->elemType() != ifFalse()->elemType()) {
            ASSERT(false, "Type not same: '"<< TensorElemTypeToStr(ifTrue()->elemType()) <<"' != '" << TensorElemTypeToStr(ifFalse()->elemType()) << "'");
        }
        elemType_ = ifTrue()->elemType();
        bool anyScattered = cond()->scattered() || ifTrue()->scattered() || ifFalse()->scattered();
        bool anyNotPointwiseOrScattered = (!cond()->isPointwise() && !cond()->scattered()) ||
                                          (!ifTrue()->isPointwise() && !ifTrue()->scattered()) ||
                                          (!ifFalse()->isPointwise() && !ifFalse()->scattered());
        ASSERT(!(anyScattered && anyNotPointwiseOrScattered), "Operation between a scattered and continuous operand not allowed");
        
        size_t dimsCond = cond()->dims();
        size_t dimsIfTrue = ifTrue()->dims();
        size_t dimsIfFalse = ifFalse()->dims();
        
        if (dimsCond != dimsIfTrue && (!cond()->isPointwise() && !ifTrue()->isPointwise())) {
            ASSERT(false, "First operand dims '"<<dimsCond<<"' do not match with second '"<<dimsIfTrue<<"'");
        }
        if (dimsIfTrue != dimsIfFalse && (!ifTrue()->isPointwise() && !ifFalse()->isPointwise())) {
            ASSERT(false, "Second operand dims '"<<dimsIfTrue<<"' do not match with third '"<<dimsIfFalse<<"'");
        }
        if (dimsIfFalse != dimsCond && (!ifFalse()->isPointwise() && !cond()->isPointwise())) {
            ASSERT(false, "Third operand dims '"<<dimsIfFalse<<"' do not match with first '"<<dimsCond<<"'");
        }
        
        #if 0
        for (size_t dim = 0; dim < std::max(std::max(dimsCond, dimsIfTrue), dimsIfFalse); dim++) {
            size_t sizeCond = cond()->isPointwise() ? cond()->size(0) : cond()->size(dim);
            size_t sizeIfTrue = ifTrue()->isPointwise() ? ifTrue()->size(0) : ifTrue()->size(dim);
            size_t sizeIfFalse = ifFalse()->isPointwise() ? ifFalse()->size(0) : ifFalse()->size(dim);
            
            if (sizeCond != sizeIfTrue && (!cond()->isPointwise() && !ifTrue()->isPointwise())) {
                ASSERT(false, "First operand dims for dim '"<< dim <<"', '"<<sizeCond<<"' do not match with second '"<<sizeIfTrue<<"'");
            }
            if (sizeIfTrue != sizeIfFalse && (!ifTrue()->isPointwise() && !ifFalse()->isPointwise())) {
                ASSERT(false, "Second operand dims for dim '"<< dim <<"', '"<<sizeIfTrue<<"' do not match with third '"<<sizeIfFalse<<"'");
            }
            if (sizeIfFalse != sizeCond && (!ifFalse()->isPointwise() && !cond()->isPointwise())) {
                ASSERT(false, "Third operand dims for dim '"<< dim <<"', '"<<sizeIfFalse<<"' do not match with first '"<<sizeCond<<"'");
            }

            dimSizes_.push_back(std::max(std::max(sizeCond, sizeIfTrue), sizeIfFalse));
        }

        //TODO remove gpus and set layout instead. 
        size_t numGPUsCond = cond()->numGPUs();
        size_t numGPUsIfTrue = ifTrue()->numGPUs();
        size_t numGPUsIfFalse = ifFalse()->numGPUs();

        if (numGPUsCond != numGPUsIfTrue && (!cond()->isConstant() && !ifTrue()->isConstant())) {
            ASSERT(false, "First operand numgpus '"<<numGPUsCond<<"' do not match with second '"<<numGPUsIfTrue<<"'");
        }
        if (numGPUsIfTrue != numGPUsIfFalse && (!ifTrue()->isConstant() && !ifFalse()->isConstant())) {
            ASSERT(false, "Second operand numgpus '"<<numGPUsIfTrue<<"' do not match with third '"<<numGPUsIfFalse<<"'");
        }
        if (numGPUsIfFalse != numGPUsCond && (!ifFalse()->isConstant() && !cond()->isConstant())) {
            ASSERT(false, "Third operand numgpus '"<<numGPUsIfFalse<<"' do not match with first '"<<numGPUsCond<<"'");
        }

        for (size_t i = 0; i < std::max(std::max(numGPUsCond, numGPUsIfTrue), numGPUsIfFalse); i++) {
            int gpuCond = cond()->isConstant() ? 0 : cond()->gpu(i);
            int gpuIfTrue = ifTrue()->isConstant() ? 0 : ifTrue()->gpu(i);
            int gpuIfFalse = ifFalse()->isConstant() ? 0 : ifFalse()->gpu(i);
            
            if (gpuCond != gpuIfTrue && (!cond()->isConstant() && !ifTrue()->isConstant())) {
                ASSERT(false, "First operand dims for gpu '"<< i <<"', '"<<gpuCond<<"' do not match with second '"<<gpuIfTrue<<"'");
            }
            if (gpuIfTrue != gpuIfFalse && (!ifTrue()->isConstant() && !ifFalse()->isConstant())) {
                ASSERT(false, "Second operand dims for gpu '"<< i <<"', '"<<gpuIfTrue<<"' do not match with third '"<<gpuIfFalse<<"'");
            }
            if (gpuIfFalse != gpuCond && (!ifFalse()->isConstant() && !cond()->isConstant())) {
                ASSERT(false, "Third operand dims for gpu '"<< i <<"', '"<<gpuIfFalse<<"' do not match with first '"<<gpuCond<<"'");
            }
            
            int gpuBinOp;

            if (!cond()->isConstant()) {
                gpuBinOp = gpuCond;
            } else if (!ifTrue()->isConstant()) {
                gpuBinOp = gpuIfTrue;
            } else if (!ifFalse()->isConstant()) {
                gpuBinOp = gpuIfFalse;
            } else {
                gpuBinOp = 0;
            }

            gpus_.push_back(gpuBinOp);
        }
        #endif
    }
};


class DropoutImpl : public ExpressionImpl {
private:
    float prob_;
public:
    DropoutImpl(std::shared_ptr<StageImpl> arg, float prob)  : 
        ExpressionImpl(AstNodeType::DropoutNode, arg, arg->group()), prob_(prob) 
    {
        setupAndCheckDimensions();
    }
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    float prob(){return prob_;}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    std::string name() {return name_;}
    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);}
    virtual size_t dims() {return arg()->dims();}
    virtual const std::vector<std::shared_ptr<ExpressionImpl>>& dimSizes() {return dimSizes_;}

    virtual void setupAndCheckDimensions() {
        dimSizes_.clear();
        layout_ = arg()->layout();
        elemType_ = arg()->elemType();
    }
};

class UpdateImpl : public ExpressionImpl {
public:
    UpdateImpl(std::shared_ptr<TensorImpl> t, std::shared_ptr<ExpressionImpl> def)  : 
        ExpressionImpl(AstNodeType::UpdateNode, t, def, nullptr)
    {
        setupAndCheckDimensions();
    }
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    std::shared_ptr<TensorImpl> arg() {return std::dynamic_pointer_cast<TensorImpl>(children_[0]);}
    std::shared_ptr<ExpressionImpl> update() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[1]);}
    
    virtual void setupAndCheckDimensions() {
        dimSizes_.clear();
        ASSERT(arg()->dimSizes() == update()->dimSizes(), "Dimensions of argument and update are different.");
        if (arg()->layout() == Replicated && update()->layout() == Sliced) {
            //This case allows us to have an implicit AllGather over the update.
            layout_ = Sliced;
        } else {
            ASSERT(arg()->layout() == update()->layout(), "Layout of argument and update are different.");
            layout_ = update()->layout();
        }
        ASSERT(arg()->elemType() == update()->elemType(), "Element type of argument and update are different.");

        for(size_t s = 0; s < update()->dims(); s++) {
            dimSizes_.push_back(update()->size(s));
        }
        
        layout_ = update()->layout();
        elemType_ = update()->elemType();

        ASSERT(arg()->group() == update()->group(), "Group of argument and update is different.");
        group_ = arg()->group();
    }
};


class MatMulImpl : public ExpressionImpl {
public:
    MatMulImpl(std::shared_ptr<ExpressionImpl> m1, std::shared_ptr<ExpressionImpl> m2) :
        ExpressionImpl(AstNodeType::MatMulNode, 
                      AstNodeImpl::asExpressionImpl(m1), 
                      AstNodeImpl::asExpressionImpl(m2), m1->group())
    {
        setupAndCheckDimensions();
    }
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> operand(int i) {return std::dynamic_pointer_cast<ExpressionImpl>(children_[i]);}
    virtual void setupAndCheckDimensions() 
    {
        //std::cout << typeid(*arg().get()).name() << std::endl;
        dimSizes_.clear();
        ASSERT(operand(0)->elemType() == operand(1)->elemType(), "Element types of operands are different.");
        elemType_ = operand(0)->elemType();
        ASSERT(operand(0)->dims() >= 2, "Operand(0) has dimensions '" << operand(0)->dims() << "' < 2");
        ASSERT(operand(1)->dims() == 2, "Operand(0) has dimensions '" << operand(1)->dims() << "' != 2");
       
        std::shared_ptr<ExpressionImpl> M1 = nullptr;
        for (int i = 0; i < operand(0)->dims() - 1; i++) {
            auto d = operand(0)->dimSizes()[i];
            dimSizes_.push_back(d);
        }
        dimSizes_.push_back(operand(1)->dimSizes()[operand(1)->dims() - 1]);

        auto N1 = operand(0)->dimSizes()[operand(0)->dims() - 1];
        auto M2 = operand(1)->dimSizes()[0];
        
        ASSERT((N1 == M2), "Last dimension of first operand != first dimension of second operaton " << "'" << N1->name() << "!= " << M2->name() << "'");

        //If any of the operands are Local then the layout is Local.
        if (operand(0)->layout() == Local || operand(1)->layout() == Local)
            layout_ = Local;
        //If both operands are Replicated then layout is replicated
        else if (operand(0)->layout() == Replicated && operand(1)->layout() == Replicated)
            layout_ = Replicated;
        else {
            //If op(0) is sliced in first dimension and other is replicated then layout is Sliced in first dimension
            if (operand(0)->layout() == Sliced && operand(1)->layout() == Replicated)
                layout_ = Sliced;
            //If op(1) is sliced in second dimension and op(0) is replicated then layout is Sliced in second dimension
            else if (operand(0)->layout() == Replicated && operand(1)->layout() == Sliced_2)
                layout_ = Sliced_2;
            //if both are sliced then layout is local
            else if ((operand(0)->layout() == Sliced || operand(0)->layout() == Sliced_2) && (operand(1)->layout() == Sliced || operand(1)->layout() == Sliced_2))
                layout_ = Local;
            else
                ASSERT(false, "Not implemented");
        }
        
        ASSERT(operand(0)->group() == operand(1)->group(), "Groups of both operands are different.");
    }
};

class ReduceTensorImpl : public ExpressionImpl {
protected:
    ReduceOperation op_;
public:
    ReduceTensorImpl(std::shared_ptr<TensorImpl> t, ReduceOperation op) : 
        ExpressionImpl(AstNodeType::ReduceTensorNode, t, t->group()), op_(op) {}
    ReduceTensorImpl(std::shared_ptr<StageImpl> s, ReduceOperation op) : 
        ExpressionImpl(AstNodeType::ReduceTensorNode, s, s->group()), op_(op) {}
    
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    
    ReduceOperation op() {return op_;}
    virtual TensorElemType elemType() {return arg()->elemType();}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return 1;
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) 
    {ASSERT(dim == 0, "Result of ReduceTensor is always of 1 dimension"); return std::shared_ptr<ExpressionImpl>(new ConstInt32(1));};
    virtual TensorLayout layout() {return arg()->layout();}

    virtual void setupAndCheckDimensions() 
    {
        //Nothing to do here
    }
};


class NormImpl : public ExpressionImpl {
public:
    NormImpl(std::shared_ptr<TensorImpl> t) : 
        ExpressionImpl(AstNodeType::NormNode, t, t->group()) {}
    NormImpl(std::shared_ptr<StageImpl> s) :
        ExpressionImpl(AstNodeType::NormNode, s, s->group()) {}
    
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    
    virtual TensorElemType elemType() {return arg()->elemType();}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {return 1;}

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) 
    {return std::shared_ptr<ExpressionImpl>(new ConstInt32(1));};
    virtual TensorLayout layout() {return arg()->layout();}

    virtual void setupAndCheckDimensions() 
    {
        //Nothing to do here
    }
};

class CastImpl : public ExpressionImpl {
private:
    TensorElemType castType_;
public:
    CastImpl(TensorElemType castType, std::shared_ptr<ExpressionImpl> op) :
        ExpressionImpl(AstNodeType::CastNode, op, op->group()), castType_(castType)
    {
        setupAndCheckDimensions();
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> op() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}

    virtual size_t dims() 
    {
        return op()->dims();
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return op()->size(dim);};
    virtual TensorLayout layout() {return op()->layout();}
    virtual TensorElemType elemType() {return castType_;}
    virtual void setupAndCheckDimensions() {
        //Nothing to do here
    }
};

// class ScatteredStageImpl : public ExpressionImpl {
// private:
//     std::string name_;
//     TensorElemType elemType_;
//     std::vector<size_t> dimSizes_;
//     std::vector<int> gpus_;

// public:
//     ScatteredStageImpl(std::shared_ptr<ExpressionImpl> definition)  : 
//         ExpressionImpl(AstNodeType::StageNode, definition) 
//     {
//         name_ = "S" + std::to_string(nameCounter++);
//         setupAndCheckDimensions();
//     }

//     virtual void accept(AstVisitor& v) {
//         v.visit(*this);
//     }

//     std::shared_ptr<ExpressionImpl> definition() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
//     std::string name() {return name_;}
//     virtual size_t dims() 
//     {
//         return definition()->dims();
//     };

//     virtual size_t size(size_t dim) {return definition()->size(dim);};
//     virtual size_t numGPUs() {return gpus_.size();}
//     virtual int gpu(int g) {return gpus_[g];}
//     virtual void setupAndCheckDimensions() {
//         for(size_t s = 0; s < definition()->dims(); s++) {
//             dimSizes_.push_back(definition()->size(s));
//         }
        
//         for (int g = 0; g < definition()->numGPUs(); g++) {
//             gpus_.push_back(definition()->gpu(g));
//         }
//     }
// };

class CommCollPrimitiveImpl : public ExpressionImpl {
protected:

public:
    CommCollPrimitiveImpl(AstNodeType n, std::shared_ptr<TensorImpl> t, std::shared_ptr<ProcessGroupIDImpl> group) :
        ExpressionImpl(n, t, group) {}
    CommCollPrimitiveImpl(AstNodeType n, std::shared_ptr<StageImpl> s, std::shared_ptr<ProcessGroupIDImpl> group) :
        ExpressionImpl(n, s, group) {}

    CommCollPrimitiveImpl(AstNodeType n, std::shared_ptr<TensorImpl> t, std::shared_ptr<ExpressionImpl> dst, std::shared_ptr<ProcessGroupIDImpl> group) :
        ExpressionImpl(n, t, dst, group) {}
    CommCollPrimitiveImpl(AstNodeType n, std::shared_ptr<StageImpl> s, std::shared_ptr<ExpressionImpl> dst, std::shared_ptr<ProcessGroupIDImpl> group) :
        ExpressionImpl(n, s, dst, group) {}
};

/*Collective Communications*/
class AllReduceImpl : public CommCollPrimitiveImpl {
protected:
    ReduceOperation op_;
public:
    //All Reduce will not work on scattered expressions
    AllReduceImpl(std::shared_ptr<TensorImpl> t, ReduceOperation op) : 
        CommCollPrimitiveImpl(AstNodeType::AllReduceNode, t, t->group()), op_(op) {
            setupAndCheckDimensions();
        }
    
    AllReduceImpl(std::shared_ptr<StageImpl> s, ReduceOperation op) : 
        CommCollPrimitiveImpl(AstNodeType::AllReduceNode, s, s->group()), op_(op) {setupAndCheckDimensions();}

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    ReduceOperation reduceOp() {return op_;}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return arg()->dims();
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);};

    virtual void setupAndCheckDimensions() {
        ASSERT(!arg()->scattered(), "Input to AllReduce should not be scattered");
        dimSizes_.clear();
        for (size_t s = 0; s < dims(); s++) {
            dimSizes_.push_back(size(s));
        }

        elemType_ = arg()->elemType();
        layout_ = Replicated;
    }
};

class FusedAllReduceImpl : public AllReduceImpl {
protected:
    std::shared_ptr<ExpressionImpl> fusedComp;
public:
    FusedAllReduceImpl(std::shared_ptr<TensorImpl> t, ReduceOperation op) : 
        AllReduceImpl(t, op) {}
    
    FusedAllReduceImpl(std::shared_ptr<StageImpl> s, ReduceOperation op) : 
        AllReduceImpl(s, op) {}

    
    virtual void setupAndCheckDimensions() {
        //FusedAllReduce is valid only when AllReduce is valid
        AllReduceImpl::setupAndCheckDimensions();
        //fusedComp should be scattered
        ASSERT(!fusedComp->scattered(), "Input to AllReduce should not be scattered");
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
};


class BroadcastImpl : public CommCollPrimitiveImpl {
public:
    //Broadcast will not work on scattered expressions
    BroadcastImpl(std::shared_ptr<TensorImpl> t, std::vector<int> gpus) : 
        CommCollPrimitiveImpl(AstNodeType::BroadcastNode, t, t->group())
    {
        /*Tensor should be stored on only one GPU, which is the root*/
        // ASSERT(t->numGPUs() == 1, "Broadcast can only broadcast a tensor stored on only one gpu. Input tensor is stored on " << t->numGPUs() << " gpus");

        setupAndCheckDimensions();
        layout_ = Replicated;
    }

    BroadcastImpl(std::shared_ptr<TensorImpl> t, int NumGPUs) : 
        BroadcastImpl(t, range(0, NumGPUs, 1)) {       
    }
    
    BroadcastImpl(std::shared_ptr<StageImpl> s, std::vector<int> gpus) : 
        CommCollPrimitiveImpl(AstNodeType::BroadcastNode, s, s->group())
    {
        setupAndCheckDimensions();
        layout_ = Replicated;    
    }

    BroadcastImpl(std::shared_ptr<StageImpl> s, int NumGPUs) : 
        BroadcastImpl(s, range(0, NumGPUs, 1)) {
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    int root() {ASSERT(false, "FIXME");}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return arg()->dims();
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);};

    virtual void setupAndCheckDimensions() {
        ASSERT(!arg()->scattered(), "Input to Broadcast should not be scattered");
        dimSizes_.clear();
        for (size_t s = 0; s < dims(); s++) {
            dimSizes_.push_back(size(s));
        }
        elemType_ = arg()->elemType();
    }
};

class ReduceImpl : public CommCollPrimitiveImpl {
protected:
    ReduceOperation op_;
    int root_;
public:
    //Reduce will never work on scattered expressions
    ReduceImpl(std::shared_ptr<TensorImpl> t, int root) : 
        CommCollPrimitiveImpl(AstNodeType::ReduceNode, t, t->group()),
        root_(root) {
        /*Only one gpu stores result of reduce*/
        setupAndCheckDimensions();
        layout_ = Local;
    }
    
    ReduceImpl(std::shared_ptr<StageImpl> s, int root) : 
        CommCollPrimitiveImpl(AstNodeType::ReduceNode, s, s->group()) {
        /*Only one gpu stores result of reduce*/
        layout_ = Local;
        setupAndCheckDimensions();
    }

    int root() {return root_;}
    ReduceOperation reduceOp() {return op_;}
    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return arg()->dims();
    };

    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);};

    virtual void setupAndCheckDimensions() {
        ASSERT(!arg()->scattered(), "Input to Reduce should not be scattered");
        dimSizes_.clear();
        for (size_t s = 0; s < dims(); s++) {
            dimSizes_.push_back(size(s));
        }
        elemType_ = arg()->elemType();
        /*Output tensor is stored on root and input tensor must also be stored on root.*/
        bool found = false;
        ASSERT(false, "FIXME Below");
        // int root = gpus_[0];
        // for (int i = 0; i < arg()->numGPUs(); i++) {
        //     if (arg()->gpu(i) == root) {
        //         found = true;
        //         break;
        //     }
        // }

        // ASSERT(found, "Input tensor is not stored on root, '" << root << "'");
    }
};

class AllGatherImpl : public CommCollPrimitiveImpl {
public:
    AllGatherImpl(std::shared_ptr<TensorImpl> t) : 
        CommCollPrimitiveImpl(AstNodeType::AllGatherNode, t, t->group())
    {
        setupAndCheckDimensions();
    }
    
    AllGatherImpl(std::shared_ptr<StageImpl> s) : 
        CommCollPrimitiveImpl(AstNodeType::AllGatherNode, s, s->group())
    {
        setupAndCheckDimensions();
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims()
    {
        return arg()->dims();
    };
    virtual TensorElemType elemType(){return arg()->elemType();}
    virtual void setupAndCheckDimensions() {
        /*Size of output tensor is number of gpus * size of input tensor*/
        /*For simplicity we multiply the size in the first dimension only*/
        ASSERT(arg()->scattered(), "Input to AllGather should be scattered");
        dimSizes_.clear();
        //Since a scattered stage's size is always (size of on each gpu)*(number of gpus), 
        //we do not need to multiply it by number of gpus.
        for (size_t s = 0; s < arg()->dims(); s++) {
            dimSizes_.push_back(arg()->size(s));
        }
        
        elemType_ = arg()->elemType();
        /*Output tensor will be stored on the same gpus as the input tensor*/
        layout_ = Replicated;
    }
};

class ReduceScatterImpl : public CommCollPrimitiveImpl {
protected:
    ReduceOperation op_;
public:
    //ReduceScatter returns a scattered expressions
    ReduceScatterImpl(std::shared_ptr<TensorImpl> t, ReduceOperation op) : 
        CommCollPrimitiveImpl(AstNodeType::ReduceScatterNode, t, t->group()), op_(op)
    {
        setupAndCheckDimensions();
        elemType_ = t->elemType();
    }
    
    ReduceScatterImpl(std::shared_ptr<StageImpl> s, ReduceOperation op) : 
        CommCollPrimitiveImpl(AstNodeType::ReduceScatterNode, s, s->group()), op_(op)
    {
        setupAndCheckDimensions();
        elemType_ = s->elemType();
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }

    ReduceOperation reduceOp() {return op_;}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() 
    {
        return arg()->dims();
    };

    virtual void setupAndCheckDimensions() {
        /*Size of output tensor is size of input tensor/number of GPUs*/
        /*For simplicity we divide the size in the first dimension only*/
        dimSizes_.clear();
        for (size_t s = 0; s < arg()->dims(); s++) {
            dimSizes_.push_back(arg()->size(s));
        }

        /*Output tensor will be stored on the same gpus as the input tensor*/
        layout_ = Sliced;
    }
};

class SendImpl : public CommCollPrimitiveImpl {
public:
    SendImpl(std::shared_ptr<TensorImpl> t, std::shared_ptr<ExpressionImpl> dst) : 
        CommCollPrimitiveImpl(AstNodeType::SendNode, t, dst, nullptr)
    {
        setupAndCheckDimensions();
    }
    
    SendImpl(std::shared_ptr<StageImpl> s, std::shared_ptr<ExpressionImpl> dst) : 
        CommCollPrimitiveImpl(AstNodeType::SendNode, s, dst, nullptr)
    {
        setupAndCheckDimensions();
    }

    SendImpl(std::shared_ptr<StageImpl> s, std::shared_ptr<ProcessGroupIDImpl> groupid,
             std::shared_ptr<ExpressionImpl> dstRank) :
        CommCollPrimitiveImpl(AstNodeType::SendNode, s, dstRank, groupid)
    {
        setupAndCheckDimensions();
    }

    SendImpl(std::shared_ptr<TensorImpl> t, std::shared_ptr<ProcessGroupIDImpl> groupid,
             std::shared_ptr<ExpressionImpl> dstRank) :
        CommCollPrimitiveImpl(AstNodeType::SendNode, t, dstRank, groupid)
    {
        setupAndCheckDimensions();
    }

    virtual void accept(AstVisitor& v) {
        v.visit(*this);
    }
    std::shared_ptr<ExpressionImpl> dstRank() {return AstNodeImpl::asExpressionImpl(children_[1]);}
    std::shared_ptr<ProcessGroupIDImpl> dstGroup() {return group();}
    std::shared_ptr<ExpressionImpl> arg() {return std::dynamic_pointer_cast<ExpressionImpl>(children_[0]);}
    virtual size_t dims() {return arg()->dims();};
    virtual TensorElemType elemType(){return arg()->elemType();}
    virtual const std::vector<std::shared_ptr<ExpressionImpl>>& dimSizes() {return arg()->dimSizes();}
    virtual std::shared_ptr<ExpressionImpl> size(size_t dim) {return arg()->size(dim);}
    virtual void setupAndCheckDimensions() {
        dimSizes_.clear();
        elemType_ = arg()->elemType();
        layout_ = arg()->layout();
    }
};
}

#endif
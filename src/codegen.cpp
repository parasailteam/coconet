#include "codegen.hpp"
#include "pipeline.hpp"
#include "astvisitor.hpp"
#include "utils.hpp"

#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <unistd.h>
#include <unordered_map>
#include <map>
#include <regex>

using namespace ACCCDSLImpl;

//Declaration and names of common variables
const std::string rankVar = "rank";
const std::string rankVarTy = "int";
const std::string rankString = "rank";
const std::string commSizeArg = "comm_size";
const std::string commSizeTy = "int";
const std::string cublasHandleVar = "cublasHandle";
const std::string cublasHandleTy = "cublasHandle_t";
const std::string threadIdxInGrid = "threadIdx.x + blockIdx.x * blockDim.x";
const std::string curandStateVar = "curandState0";

class NumElemGen : public AstVisitor
{
    //A Simple visitor to generate the expression for number of elements
    private:
        std::stringstream& os_;
    public:
        NumElemGen(std::stringstream& ss) : os_(ss) {}

        void print(ExpressionImpl& node) {
            node.accept(*this);
        }

        void visit(TensorImpl& node) {
        }

        void visit(AllReduceImpl& node) {
        }

        virtual void visit(ReduceImpl& node) 
        {
        }

        virtual void visit(BroadcastImpl& node) 
        {
        }

        virtual void visit(AllGatherImpl& node) 
        {
        }
        virtual void visit(ReduceScatterImpl& node) 
        {
        }

        void visit(BinaryPointwiseOp& node) {
            os_ << "(";
            node.operand(0)->accept(*this);
            os_ << " " << BinaryPointwiseOp::operatorToStr(node.op()) << " ";
            node.operand(1)->accept(*this);
            os_ << ")";
        }

        void visit(CastImpl& node) {
        }
        void visit(UnaryPointwiseOp& node) {
            os_ << UnaryPointwiseOp::operatorToStr(node.op()) << "(";
            node.operand()->accept(*this);
            os_ << ")";
        }
        void visit(PowerImpl& node) {
            visitChildren(node);
        }
        void visit(ReduceTensorImpl& node) {
            visitChildren(node);
        }
        void visit(MatMulImpl& node) {
            ASSERT(false, "TODO: Implement");
            visitChildren(node);
        }

        void visit(StageImpl& node) {
        }
        virtual void visit(DropoutImpl& node)
        {
            visitChildren(node);
        }
        void visit(NormImpl& node) {
        }
        virtual void visit(ScatterImpl& node) {
            visitChildren(node);
        }

        virtual void visit(IteImpl& node) {
            ASSERT(false, "To implement");
        }

        void visit(UpdateImpl& node) {
            ASSERT(false, "to implement");
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
            os_ << "(" << node.val() << ")";
        }
        virtual void visit(ConstFloat32& node) {
            os_ << node.val();
        }
        virtual void visit(ConstFloat64& node) {
            os_ << node.val();
        }  
};

//Print from startDim to endDim (not including)
std::string genNumElem(std::shared_ptr<ExpressionImpl> numElemExpr, int startDim, int endDim)
{
    std::stringstream numElemStream;
    if (numElemExpr->layout() == Sliced || numElemExpr->layout() == Sliced_2)
        numElemStream << "DIVUP(";

    if (numElemExpr->dimSizes().size() == 1) {
        NumElemGen numElemGen(numElemStream);
        numElemGen.print(*numElemExpr->dimSizes()[0].get());
    } else {
        numElemStream << "(";
        for (auto iter = numElemExpr->dimSizes().begin(); iter != numElemExpr->dimSizes().end();) {
            if (iter - numElemExpr->dimSizes().begin() < startDim) {
                iter++;
                continue;
            }
            if (iter - numElemExpr->dimSizes().begin() >= endDim)
                break;
            NumElemGen numElemGen(numElemStream);
            numElemGen.print(*(*iter).get());   
            iter++;
            if (iter - numElemExpr->dimSizes().begin() != endDim) {
                numElemStream << "*";
            }
        }
        numElemStream << ")";
    }
    
    if (numElemExpr->layout() == Sliced || numElemExpr->layout() == Sliced_2)
        numElemStream << ", comm_size)";

    return numElemStream.str();
}

std::string genNumElem(std::shared_ptr<ExpressionImpl> numElemExpr)
{
    return genNumElem(numElemExpr, 0, numElemExpr->dimSizes().size());
}

template<typename T>
std::set<std::shared_ptr<ExpressionImpl>> allDimExprs(T inputIt, T outIt)
{
    std::set<std::shared_ptr<ExpressionImpl>> dimExprs;

    for (auto it = inputIt; it != outIt; it++) {
        for (auto dimSize : (*it)->dimSizes()) {
            if (dimSize->type() == VariableNode)
                //Only Variables can be dimension sizes
                dimExprs.insert(dimSize);
        }
    }

    return dimExprs;
}

ReduceOperation getCommCollRedOp(AstNodeImpl& node) 
{
    switch (node.type()) {
        case AllReduceNode:
        {
            return ((AllReduceImpl&)node).reduceOp();
        }
        case ReduceNode:
        {
            return ((ReduceImpl&)node).reduceOp();
        }
        case ReduceScatterNode:
        {
            return ((ReduceScatterImpl&)node).reduceOp();
        }

        default:
            return ReduceOperationNone;
    }
}

std::string uintTypeForSize(const size_t sz)
{
    switch (sz) {
        case 1:
            return "uint8_t";
        case 2:
            return "uint16_t";
        case 4:
            return "uint32_t";
        case 8:
            return "uint64_t";
        
        default:
            ASSERT(false, "Unknown int type for size " << sz);
            return "";
    }
}

std::string f16ToTypeConvCUDAFunc(TensorElemType t) 
{
    switch (t) {
        case Float32:
            return "(float)";
        // case Float64:
        // case Int16:
        // case Int32:
        default:
        //TODO: Add other conversion functions
            ASSERT(false, "TODO: Add conversion from half to other");
    }
}

std::string readFile(std::string filepath) 
{
    std::ifstream ifs(filepath.c_str(), std::ifstream::in);
    if (!ifs.is_open()) {
        ASSERT(false, "Cannot open file '"<<filepath <<"'");
    }
    std::ostringstream sstr;
    sstr << ifs.rdbuf();
    ifs.close();
    return sstr.str();
}

void writeFile(std::string filepath, const std::string& contents) 
{
    std::ofstream file(filepath.c_str(), std::ofstream::out);
    if (!file.is_open()) {
        ASSERT(false, "Cannot open file '" << filepath << "'");
    }

    file.write(contents.c_str(), contents.size());
    file.close();
}

void writeFile(int fd, const std::string& contents) 
{
    write(fd, contents.c_str(), contents.size());
    close(fd);
}

std::string exec(const std::string& cmd) 
{
    std::array<char, 128> buffer;
    std::string result;

    auto pipe = popen(cmd.c_str(), "r"); // get rid of shared_ptr

    if (!pipe) throw std::runtime_error("popen() failed!");

    while (!feof(pipe)) {
        if (fgets(buffer.data(), 128, pipe) != nullptr)
            result += buffer.data();
    }

    auto rc = pclose(pipe);

    if (rc == EXIT_SUCCESS) { // == 0

    } else {  // EXIT_FAILURE is not used by all programs, maybe needs some adaptation.
        std::cout << "executing '" << cmd << "' failed with " << rc << std::endl;
        std::cout << "output " << result << std::endl;
        ASSERT(false, "");
    }
    return result;
}

uint32_t nextPowerOf2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

uint64_t nextPowerOf2(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;

    return v;
}

uint64_t isPowerOf2(uint64_t num) {
    return ((num != 0) && ((num &(num - 1)) == 0));
}

uint64_t currOrNextPowerOf2(uint64_t num) {
    if (isPowerOf2(num))
        return num;
    return nextPowerOf2(num);
}

void replaceAllSubStringInFile(std::string filepath, std::string regexSub, std::string replacement)
{
    std::regex e(regexSub);
    std::string contents = readFile(filepath);
    contents = std::regex_replace(contents, e, replacement);
    writeFile(filepath, contents);
}

std::string printCudaOccupancyMaxActiveBlocksPerMultiprocessor(std::string blockspersm, std::string funcname, std::string threads, int shmem)
{
    return "cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&" + blockspersm + ", (void*)" + funcname + ", " + threads + ", " + std::to_string(shmem) + ")";
}

std::string printCudaLaunchCooperativeKernel(std::string funcname, std::string blocks, std::string threads, std::string args, int shmem, std::string stream)
{
  return "cudaLaunchCooperativeKernel((void*)"+funcname+", "+ blocks +", " + threads + ", " + args + ", " + std::to_string(shmem) + ", " + stream +")";
}


std::string cudaCheck(std::string s)
{
    return "CUDACHECK(" + s + ");";
}

std::string ncclCheck(std::string s)
{
    return "NCCLCHECK(" + s + ");";
}

std::string cublasCheck(std::string s)
{
    return "CUBLASCHECK(" + s + ");";
}

std::string printCUDAMalloc(std::shared_ptr<ExpressionImpl> arg, bool x=false)
{
    const std::string cudaMalloc = "cudaMalloc";
    std::stringstream code;

    code << cudaMalloc << "(&" << arg->name() << ", " << genNumElem(arg) << (x && arg->layout() == Sliced ? "*comm_size":"")<< " * sizeof(" << 
        elemTypeToCType(arg->elemType()) << "))";
    
    return cudaCheck(code.str());
}

std::string printCUDAFree(std::shared_ptr<ExpressionImpl> arg) 
{
    const std::string cudaFree = "cudaFree";
    std::stringstream code;

    code << cudaFree << "(" << arg->name() << ")";
    
    return cudaCheck(code.str());
}

std::string printCUDAMemcpyD2H(std::shared_ptr<ExpressionImpl> dst, std::shared_ptr<ExpressionImpl> src) 
{
    const std::string cudaFree = "cudaMemcpy";
    std::stringstream code;

    code << cudaFree << "(" << dst->name() << ", " << src->name() << ", " << 
        genNumElem(dst) << "*" << "sizeof(" << elemTypeToCType(dst->elemType()) << "), cudaMemcpyDeviceToHost" << ")";
    
    return cudaCheck(code.str());
}

std::string printCUDAMemcpyHalfD2FloatH(std::shared_ptr<ExpressionImpl> dst, std::shared_ptr<ExpressionImpl> src) 
{
    const std::string cudaFree = "cudaMemcpyHalfDevice2FloatHost";
    std::stringstream code;

    code << cudaFree << "(" << dst->name() << ", " << src->name() << ", " << 
        genNumElem(dst) << ")";
    
    return code.str();
}

std::string printDeclaration(std::shared_ptr<ExpressionImpl> arg, std::string endChar = ";", std::string varnamePrefix="", 
                             bool isRestrict = false, bool printTypeAsTemplate = false, TensorElemType customType = None)
{
    std::stringstream ss;
    if (!printTypeAsTemplate) {
        if (customType != None) 
            ss << elemTypeToCType(customType);
        else
            ss << elemTypeToCType(arg->elemType());
    }
    else 
        ss << "T";
    if (arg->type() == TensorNode || arg->type() == StageNode)
        ss << "* " << ((isRestrict) ? "__restrict__ " : "");
    else
        ss << " ";
    ss << varnamePrefix+arg->name() << endChar;

    return ss.str();
}

std::string printArgument(std::shared_ptr<ExpressionImpl> arg, TensorElemType customType = None)
{
    return printDeclaration(arg, "", "", false, false, customType);
}

std::string printDeclarationForNCCLPrims(std::shared_ptr<ExpressionImpl> arg, bool isRestrict=false) 
{
    //Print the declaration as a template T
    return printDeclaration(arg, ";", "", isRestrict, true);
}

std::string printArgumentForNCCLPrims(std::shared_ptr<ExpressionImpl> arg)
{
    return printDeclaration(arg, "", "", false, true);
}

std::string printNew(std::shared_ptr<ExpressionImpl> arg, std::string varnamePrefix="", TensorElemType customType = None)
{
    std::stringstream ss;

    ss << varnamePrefix+arg->name() << " = new "
       << (customType != None ? elemTypeToCType(customType) : elemTypeToCType(arg->elemType())) << "[" << genNumElem(arg) << "];";

    return ss.str();
}

std::string printEventCreate(std::string eventName) 
{
      return "CUDACHECK(cudaEventCreate(&" + eventName +"));";
}

std::string printEventRecord(std::string eventName, std::string cudaStream) 
{
      return "CUDACHECK(cudaEventRecord(" + eventName +", " + cudaStream + "));";
}

std::string printEventSynchronize(std::string eventName) 
{
      return "CUDACHECK(cudaEventSynchronize(" + eventName +"));";
}

std::string printEventElapsedTime(std::string eventName1, std::string eventName2, std::string timeVar) 
{
      return "CUDACHECK(cudaEventElapsedTime(&" + timeVar + ", " + eventName1 + "," + eventName2 + "));";
}

static std::string indent(int level) 
{
    std::string s = "";
    for (int i = 0; i < level; i++) {
        s += "  ";
    }
    return s;
}

std::string iteratorForDim(size_t dim) 
{
    return "i" + std::to_string(dim);
}


std::string iteratorAccessString(std::shared_ptr<ExpressionImpl> node)
{
    std::string s = "";
    
    for (size_t i = 0; i < node->dims(); i++) {
        auto iterAccess = iteratorForDim(i);
        auto numElem = (i != node->dims() - 1) ? "*"+genNumElem(node, i+1, node->dims()): "";
        s += iterAccess + numElem;
        if (i != node->dims() - 1)
            s += "+";
    }

    return s;
}

std::string iteratorAccessPrintfFormat(size_t ndims) 
{
    std::string s = "";

    for (size_t i = 0; i < ndims; i++) {
        s += "%ld";
        if (i != ndims - 1)
            s += ", ";
    }

    return s;
}

std::string iteratorAccessPrintfString(size_t ndims) 
{
    std::string s = "";

    for (size_t i = 0; i < ndims; i++) {
        s += iteratorForDim(i);
        if (i != ndims - 1)
            s += ", ";
    }

    return s;
}

std::string iteratorInit(size_t ndims, std::string indentStr)
{
    std::string s = "";
    if (ndims >= 1) 
        s += indentStr + "int i0 = threadIdx.x + blockDim.x*blockIdx.x;";
    
    if (ndims >= 2)
        s += indentStr + "int i1 = threadIdx.y + blockDim.y*blockIdx.y;";

    if (ndims == 3)
        s += indentStr + "int i2 = threadIdx.z + blockDim.z*blockIdx.z;";
    
    ASSERT(ndims <= 3, "CUDA Codegeneration not supported for Tensors with dims > 3");

    return s;
}

enum CodeType 
{
    MPI,
    CUDA
};

class PointwiseOpCodegen : public AstVisitor
{
    private:
        std::stringstream& os_;
        PipelineStage* pipeStage_;
        Pipeline& pipeline_;
        bool generateCheck_;
        bool generateAsVars_;
        TensorElemType explicitType_;
        bool useHalf2;
        std::vector<std::shared_ptr<CastImpl>> mixedPrecisionCasts_;
        CodeType codeType_;
        std::stringstream declarations;
        bool genNumpyBroadcast;

    public:
        PointwiseOpCodegen(std::stringstream& os, PipelineStage* pipeStage, Pipeline& pipeline, bool generateCheck,
                                 bool generateAsVars, CodeType codeType, std::vector<std::shared_ptr<CastImpl>> mixedPrecisionCasts) : useHalf2(false), genNumpyBroadcast(false), explicitType_(None), os_(os),
                                 pipeStage_(pipeStage), pipeline_(pipeline), generateCheck_(generateCheck), codeType_(codeType),
                                 generateAsVars_(generateAsVars), mixedPrecisionCasts_(mixedPrecisionCasts) {}
        PointwiseOpCodegen(std::stringstream& os, Pipeline& pipeline, bool generateCheck,
                                 CodeType codeType, std::vector<std::shared_ptr<CastImpl>> mixedPrecisionCasts) : 
            os_(os), pipeStage_(nullptr), pipeline_(pipeline), generateCheck_(generateCheck), 
            useHalf2(false), genNumpyBroadcast(false), explicitType_(None), codeType_(codeType), mixedPrecisionCasts_(mixedPrecisionCasts) {}
        PointwiseOpCodegen(std::stringstream& os, std::string iterator, Pipeline& pipeline, bool generateCheck, 
                                 CodeType codeType, std::vector<std::shared_ptr<CastImpl>> mixedPrecisionCasts) : 
            os_(os), pipeStage_(nullptr), pipeline_(pipeline), generateCheck_(generateCheck), 
            useHalf2(false), genNumpyBroadcast(false), explicitType_(None), codeType_(codeType), mixedPrecisionCasts_(mixedPrecisionCasts) {}

        void print(ExpressionImpl& node) {
            node.accept(*this);
        }

        std::string decls() {return declarations.str();}

        void setExplicitType(TensorElemType t) {
            explicitType_ = t;
        }

        void setHalf2Type() {
            useHalf2 = true;
        }

        void visit(TensorImpl& node) {
            std::shared_ptr<TensorImpl> shptr = pipeline_.sharedPtrForAstPtr(&node);
            os_ << ((generateCheck_ ? "__" : "") + node.name());
            if (!generateAsVars_) {
                os_ << "[";
                os_ << iteratorForDim(0);
                if (genNumpyBroadcast)
                    os_ << "%" << genNumElem(shptr);
                os_ << "]";
            }
        }

        void visit(AllReduceImpl& node) {
        }

        virtual void visit(ReduceImpl& node) 
        {
        }

        virtual void visit(BroadcastImpl& node) 
        {
        }

        virtual void visit(AllGatherImpl& node) 
        {
        }
        virtual void visit(ReduceScatterImpl& node) 
        {
        }
        void visit(BinaryPointwiseOp& node) {
            os_ << "(";
            if (codeType_ == CodeType::CUDA && (explicitType_ == TensorElemType::Float16)) {
                ASSERT(false, "To imlement numpy-broadcast.");
                std::stringstream os0;
                PointwiseOpCodegen codegen0(os0, pipeStage_, pipeline_, generateCheck_, generateAsVars_, codeType_, mixedPrecisionCasts_);
                codegen0.setExplicitType(explicitType_);
                if (useHalf2)
                    codegen0.setHalf2Type();
                node.operand(0)->accept(codegen0);

                std::stringstream os1;
                PointwiseOpCodegen codegen1(os1, pipeStage_, pipeline_, generateCheck_, generateAsVars_, codeType_, mixedPrecisionCasts_);
                codegen1.setExplicitType(explicitType_);
                if (useHalf2)
                    codegen1.setHalf2Type();
                node.operand(1)->accept(codegen1);
                
                os_ << (useHalf2 ? BinaryPointwiseOp::operatorToHalf2Func(node.op()) : BinaryPointwiseOp::operatorToHalfFunc(node.op()))
                    << "(" << os0.str() << ", " << os1.str() << ")";
            } else {
                genNumpyBroadcast = false;
                if (node.operand(0)->dims() < node.operand(1)->dims())
                    genNumpyBroadcast = true;
                node.operand(0)->accept(*this);
                os_ << " " << BinaryPointwiseOp::operatorToStr(node.op()) << " ";
                if (node.operand(1)->dims() < node.operand(0)->dims())
                    genNumpyBroadcast = true;
                node.operand(1)->accept(*this);
            }
            os_ << ")";
        }

        void visit(CastImpl& node) {
            if (node.op()->elemType() == Float16) {
                if (codeType_ == CodeType::MPI) {
                    //Since everything is already Float32 in MPI code there
                    //is no need to the transformation 
                    if (node.elemType() != Float32) {
                        os_ << "(" << elemTypeToCType(node.elemType());
                        os_ << ")";
                    }
                    visitChildren(node);
                } else {
                    if (useHalf2 || explicitType_ == Float16)
                        visitChildren(node);
                    else {
                        os_ << f16ToTypeConvCUDAFunc(node.elemType());
                        os_ << "(";
                        visitChildren(node);
                        os_ << ")";
                    }
                }
            } else {
                os_ << "(" << elemTypeToCType(node.elemType());
                os_ << ")";
                visitChildren(node);
            }
        }
        void visit(UnaryPointwiseOp& node) {
            os_ << "(";
            if (codeType_ == CodeType::CUDA && (explicitType_ == TensorElemType::Float16)) {
                os_ << (useHalf2 ? UnaryPointwiseOp::operatorToHalf2Func(node.op()) : UnaryPointwiseOp::operatorToHalfFunc(node.op()));
                os_ << "(";
                node.operand()->accept(*this);
                os_ << ")";
            } else {
                os_ << UnaryPointwiseOp::operatorToStr(node.op()) << "(";
                node.operand()->accept(*this);
                os_ << ")";
            }
            os_ << ")";
        }
        void visit(PowerImpl& node) {
            visitChildren(node);
        }
        void visit(ReduceTensorImpl& node) {
            visitChildren(node);
        }
        void visit(NormImpl& node) {
            visitChildren(node);
        }
        void visit(MatMulImpl& node) {
            ASSERT(false, "TODO: Implement");
            visitChildren(node);
        }

        void visit(StageImpl& node) {
            std::shared_ptr<StageImpl> shptr = pipeline_.sharedPtrForAstPtr(&node);
            std::shared_ptr<ExpressionImpl> storageLoc;
            if (pipeline_.explicitStoreLocations().count(shptr) == 1 && codeType_ == CUDA)
                storageLoc = pipeline_.explicitStoreLocations().at(shptr);
            else
                storageLoc = shptr;

            auto name = storageLoc->name();
            os_ << ((generateCheck_ ? "__" : "") + name);
            if ((pipeStage_ == nullptr || pipeStage_->getStorageLocation(shptr) == Memory) &&
                !generateAsVars_) {
                os_ << "[";
                if (storageLoc != shptr && shptr->layout() == Sliced && storageLoc->layout() != Sliced) {
                    os_ << genNumElem(shptr) << " * " << rankVar << " + ";
                }
                if (node.isPointwise())
                    os_ << "0";
                else
                    os_ << iteratorForDim(0);
                if (genNumpyBroadcast)
                    os_ << "%" << genNumElem(shptr);
                os_ << "]";
            }
        }
        virtual void visit(DropoutImpl& node)
        {
            declarations << "curandState " << curandStateVar << ";" << std::endl;
            declarations << "curand_init(0, 0, 0, &" << curandStateVar << ");" << std::endl;
            os_ << "(curand_uniform(&" << curandStateVar << ") < " << node.prob() << ") ? ";
            visitChildren(node);
            os_ << " : (" << elemTypeToCType(node.elemType()) << ") 0";
        }
        virtual void visit(ScatterImpl& node) {
            visitChildren(node);
        }

        virtual void visit(IteImpl& node) {
            if (codeType_ == CodeType::CUDA && (explicitType_ == TensorElemType::Float16) && useHalf2) {
                // For half2 implement ite with multiplication and addition
                os_ << "__hadd2(__hmul2(";
                node.cond()->accept(*this);
                os_ << ",";
                node.ifTrue()->accept(*this);
                os_ << "),__hmul2(__hsub2(__half2half2(1),";
                // TODO: maybe avoid evaluating cond twice. might be okay due to CSE though
                node.cond()->accept(*this);
                os_ << "),";
                node.ifFalse()->accept(*this);
                os_ << "))";
            } else {
                os_ << "(";
                node.cond()->accept(*this);
                os_ << " ? ";
                node.ifTrue()->accept(*this);
                os_ << " : ";
                node.ifFalse()->accept(*this);
                os_ << ")";
            }
        }

        void visit(UpdateImpl& node) {
            ASSERT(false, "to implement");
        }

        void visit(VariableImpl& node) {
            // if (useHalf2) {
            //     os_ << "__half2half2(";
            // }
            
            os_ << ((generateCheck_ ? "__" : "") + node.name());

            // if (useHalf2) {
            //     os_ << ")";
            // }
        }

        virtual void visit(ConstUInt64& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__ull2half_rd(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }
        virtual void visit(ConstInt64& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__ll2half_rd(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }
        virtual void visit(ConstUInt32& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__int2half_rd(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }
        virtual void visit(ConstInt32& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__int2half_rd(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }
        virtual void visit(ConstFloat16& node) {
            if (codeType_ == CodeType::CUDA) {
                if (useHalf2) {
                    os_ << "__half2half2(";
                }
                os_ << "__float2half";
            }

            os_ << "(" << node.val() << ")";
            if (codeType_ == CodeType::CUDA) {
                if (useHalf2) {
                    os_ << ")";
                }
            }
        }
        virtual void visit(ConstFloat32& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__float2half(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }
        virtual void visit(ConstFloat64& node) {
            if (useHalf2) {
                os_ << "__half2half2(";
            }
            if (explicitType_ == TensorElemType::Float16) {
                os_ << "__double2half_rd(";
            }
            os_ << node.val();
            if (explicitType_ == TensorElemType::Float16) {
                os_ << ")";
            }
            if (useHalf2) {
                os_ << ")";
            }
        }  
};

struct ArrayDecl {
    std::string name;
    std::vector<std::shared_ptr<ExpressionImpl>> size;
    bool isCUDAArray;
};

struct CFunc {
    std::string name;
    std::string body;
    std::set<std::shared_ptr<ExpressionImpl>> arguments;
    bool isCUDA;
    AstNodeType type;
    bool useCooperativeGrid;
    std::set<std::shared_ptr<StageImpl>> intermediates;
};

struct CStruct {
    std::string name;
    std::string body;
    bool isCUDA;
};

std::vector<std::string> iteratorsForDims(size_t ndims)
{
    std::vector<std::string> s;

    for (size_t i = 0; i < ndims; i++) {
        s.push_back(iteratorForDim(i));
    }

    return s;
}

std::vector<std::shared_ptr<CastImpl>> isMixedPrecision(std::shared_ptr<ExpressionImpl> binOpNode)
{
    AllCastOpsVisitor castOpVisitor;
    std::set<std::shared_ptr<CastImpl>> castOps = binOpNode->childrenOfType<CastImpl>();
    std::vector<std::shared_ptr<CastImpl>> typeConvs;

    for (auto castOp : castOps) {
        if (sizeOfElemType(castOp->elemType()) > sizeOfElemType(castOp->op()->elemType()) and 
            castOp->elemType() != castOp->op()->elemType()) {
            typeConvs.push_back(castOp);
        }
    }

    for (auto convs : typeConvs) {
        ASSERT(convs->elemType() == Float32 && convs->op()->elemType() == Float16, "Only type conversion between f16 to f32 is supported.");
    }

    return typeConvs;
}

std::string generateReduceTensorCodeCPU(Pipeline& pipeline, std::shared_ptr<StageImpl> output, ReduceTensorImpl* reduceTensor, 
                                        bool generateCheck)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string funcName = "reduceFunc" + std::to_string(nameCounter++);
    std::set<std::shared_ptr<ExpressionImpl>> inputs;

    //Add output to arguments too.
    inputs.insert(output);
    inputs.insert(reduceTensor->arg());

    //Copy output to host if we need to generate check
    if (generateCheck) {
        std::shared_ptr<ExpressionImpl> explicitOutput;
        if (pipeline.explicitStoreLocations().count(output) > 0) {
            ASSERT(false,"Not implemented");
            // explicitOutput = pipeline.explicitStoreLocations().at(output);
        } else {
            explicitOutput = output;
        }
        auto xx = output->copyWithNewName("h" + output->name());
        std::shared_ptr<StageImpl> hostOutput = std::shared_ptr<StageImpl>(&xx);
        TensorElemType customType =  (hostOutput->elemType() == Float16) ? Float32 : hostOutput->elemType();
        codeStream << indent(1) << printDeclaration(hostOutput, ";", "", false, false, customType) << std::endl;
        codeStream << indent(1) << printNew(hostOutput, "", customType) << std::endl;
        if (hostOutput->elemType() == Float16) {
            codeStream << indent(1) << printCUDAMemcpyHalfD2FloatH(hostOutput, explicitOutput) << std::endl;
        } else {
            codeStream << indent(1) << printCUDAMemcpyD2H(hostOutput, explicitOutput) << std::endl;
        }
    }
    //Generate for loops
    int indentLevel = 1;
    for (size_t i = 0; i < reduceTensor->arg()->dims(); i++) {
        auto it = iteratorForDim(i);
        codeStream << indent(indentLevel) << "for (size_t " << it << " = 0; " << it << " < " << reduceTensor->arg()->size(i) << "; " << 
            it << "++) {" << std::endl;
        indentLevel++;
    }
    
    //If we are checking the output, i.e., generateCheck is true, then
    //we prefix each stage name with "__"    
    std::string accessStr = "[" + iteratorAccessString(output) + "]" ;
    //Print assignment to output stage
    codeStream << indent(indentLevel) << ("__" + output->name() + "[0]")
               << " = " << ("__" + output->name() + "[0]");
               
    switch (reduceTensor->op()) {
        case Summation:
            codeStream << "+";
            break;
        case Maximum:
        default:
            ASSERT(false, "Not implemented.");
    }
    codeStream << "__" << reduceTensor->arg()->name() <<accessStr << ";" << std::endl;

    if (generateCheck) {
        iteratorAccessString(output);
        std::string u = ("__" + output->name()) + accessStr;
        std::string v = "h" + output->name() + accessStr;
        codeStream << indent(indentLevel) << "if (!eqFloat(" << u << ", " << v << ")) {" << std::endl;
        codeStream << indent(indentLevel+1) << "printf(\"Mismatch at " << iteratorAccessPrintfFormat(output->dims()) << " : ref '%f', computed '%f'\\n\"," << iteratorAccessPrintfString(output->dims()) << ", " << u << ", " << v<<");"<<std::endl;
        codeStream << indent(indentLevel+1) << "return false;" << std::endl;
        codeStream << indent(indentLevel) << "}" << std::endl;
    }
    for (size_t i = 0; i < reduceTensor->arg()->dims(); i++) {
        codeStream << indent(indentLevel - 1) << "}" << std::endl;
        indentLevel--;
    }

    return codeStream.str();
}

std::string generateOpCodeCPU(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<ExpressionImpl> binOpNode, 
                                 bool generateCheck)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string binOpFuncName = "binOpFunc" + std::to_string(nameCounter++);
    std::set<std::shared_ptr<ExpressionImpl>> binOpInputs = binOpNode->usedExprs();

    //Add output to arguments too.
    binOpInputs.insert(output);
    
    //Copy output to host if we need to generate check
    if (generateCheck) {
        std::shared_ptr<ExpressionImpl> explicitOutput;
        if (pipeline.explicitStoreLocations().count(output) > 0) {
            explicitOutput = pipeline.explicitStoreLocations().at(output);
        } else {
            explicitOutput = output;
        }
        std::shared_ptr<StageImpl> hostOutput = std::shared_ptr<StageImpl>(new StageImpl(output->copyWithNewName("h" + output->name())));
        TensorElemType customType =  (hostOutput->elemType() == Float16) ? Float32 : hostOutput->elemType();
        codeStream << indent(1) << printDeclaration(hostOutput, ";", "", false, false, customType) << std::endl;
        codeStream << indent(1) << printNew(hostOutput, "", customType) << std::endl;
        if (hostOutput->elemType() == Float16) {
            codeStream << indent(1) << printCUDAMemcpyHalfD2FloatH(hostOutput, explicitOutput) << std::endl;
        } else {
            codeStream << indent(1) << printCUDAMemcpyD2H(hostOutput, explicitOutput) << std::endl;
        }
    }
    //Generate for loops
    int indentLevel = 1;
    for (size_t i = 0; i < binOpNode->dims(); i++) {
        auto it = iteratorForDim(i);
        codeStream << indent(indentLevel) << "for (size_t " << it << " = 0; " << it << " < " << binOpNode->size(i) << "; " << 
            it << "++) {" << std::endl;
        indentLevel++;
    }
    
    //If we are checking the output, i.e., generateCheck is true, then
    //we prefix each stage name with "__"

    //Print Binary operation
    std::stringstream binopCodeStream;
    PointwiseOpCodegen binOpCodegen(binopCodeStream, pipeline, true, CodeType::MPI, isMixedPrecision(binOpNode));
    binOpCodegen.print(*binOpNode);
    
    std::string accessStr = "[" + iteratorAccessString(output) + "]" ;
    //Print assignment to output stage
    codeStream << indent(indentLevel) << ("__" + output->name()) << accessStr
               << " = " << binopCodeStream.str() << ";" << std::endl;

    if (generateCheck) {
        iteratorAccessString(output);
        std::string u = ("__" + output->name()) + accessStr;
        std::string v = "h" + output->name() + accessStr;
        codeStream << indent(indentLevel) << "if (!eqFloat(" << u << ", " << v << ")) {" << std::endl;
        codeStream << indent(indentLevel+1) << "printf(\"Mismatch at " << iteratorAccessPrintfFormat(output->dims()) << " : ref '%f', computed '%f'\\n\"," << iteratorAccessPrintfString(output->dims()) << ", " << u << ", " << v<<");"<<std::endl;
        codeStream << indent(indentLevel+1) << "return false;" << std::endl;
        codeStream << indent(indentLevel) << "}" << std::endl;
    }
    for (size_t i = 0; i < binOpNode->dims(); i++) {
        codeStream << indent(indentLevel - 1) << "}" << std::endl;
        indentLevel--;
    }

    return codeStream.str();
}

CFunc generateReduceCUDA(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<ReduceTensorImpl> reduceTensor)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string funcName = "redOpFunc" + std::to_string(nameCounter++);    

    codeStream << "__global__ void " << funcName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    std::set<std::shared_ptr<ExpressionImpl>> redOpInputs;
    redOpInputs.insert(reduceTensor->arg());
    redOpInputs.insert(output);

    ASSERT(false, "FIX");
    //FIX
    // for (auto it : pipeline.explicitStoreLocations()) {
    //     redOpInputs.erase(it.first);
    //     redOpInputs.insert(it.second);
    // }

    int ii = 0;
    for (auto iter : redOpInputs) {
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        if (ii != redOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }


    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx

    codeStream << iteratorInit(reduceTensor->arg()->dims(), indent(1)) << std::endl;
    
    //Print assignment to output stage
    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : pipeline.explicitStoreLocations().at(output)->name();
    std::string inputName;
    if (reduceTensor->arg()->type() == TensorNode || reduceTensor->arg()->type() == VariableNode) {
        inputName = reduceTensor->arg()->name();
    } else {
        std::shared_ptr<StageImpl> inputStage = AstNodeImpl::asStageImpl(reduceTensor->arg());
        inputName = pipeline.explicitStoreLocations().count(inputStage) == 0 ? reduceTensor->arg()->name() : pipeline.explicitStoreLocations().at(inputStage)->name();
    }

    codeStream << indent(1);
    //TODO: Improve this by using NVIDIA CUB, maybe?
    switch (reduceTensor->op()) {
        case Summation:
            codeStream << "__atomicAdd(" << name << ", " << inputName << "[" << iteratorAccessString(output) << "]);" << std::endl;
            break;
        case Maximum:
        case Multiplication:
        case Minimum:
        default:
            ASSERT(false, "Reduction operation not implemented.");
    }

    codeStream << "}";

    return CFunc({funcName, codeStream.str(), redOpInputs, true, ReduceTensorNode});
}

CFunc generateCUBLASMatMul(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<MatMulImpl> matmul)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string funcName = "matMul" + std::to_string(nameCounter++);

    codeStream << "void " << funcName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    std::set<std::shared_ptr<ExpressionImpl>> inputs;
    inputs.insert(matmul->operand(0));
    inputs.insert(matmul->operand(1));
    
    for (auto it : pipeline.explicitStoreLocations()) {
        inputs.erase(it.first);
        inputs.insert(it.second);
    }
    
    auto dimExprs = allDimExprs(inputs.begin(), inputs.end());
    inputs.insert(dimExprs.begin(), dimExprs.end());

    int ii = 0;
    for (auto iter : inputs) {
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        codeStream << ", ";
    }
    
    codeStream << cublasHandleTy << " " << cublasHandleVar << ", " << commSizeTy << " " << commSizeArg << ", " << rankVarTy << " " << rankVar;
    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx
    
    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : pipeline.explicitStoreLocations().at(output)->name();
    std::string input1Name, input2Name;

    if (matmul->operand(0)->type() == TensorNode) {
        input1Name = matmul->operand(0)->name();
    } else {
        std::shared_ptr<StageImpl> inputStage = AstNodeImpl::asStageImpl(matmul->operand(0));
        input1Name = pipeline.explicitStoreLocations().count(inputStage) == 0 ? matmul->operand(0)->name() : pipeline.explicitStoreLocations().at(inputStage)->name();
    }

    if (matmul->operand(1)->type() == TensorNode) {
        input2Name = matmul->operand(1)->name();
    } else {
        std::shared_ptr<StageImpl> inputStage = AstNodeImpl::asStageImpl(matmul->operand(1));
        input2Name = pipeline.explicitStoreLocations().count(inputStage) == 0 ? matmul->operand(1)->name() : pipeline.explicitStoreLocations().at(inputStage)->name();
    }

    std::string cublasTypeA = elemTypeToCUBLASType(matmul->operand(0)->elemType());
    std::string cublasTypeB = elemTypeToCUBLASType(matmul->operand(1)->elemType());
    std::string cublasTypeC = elemTypeToCUBLASType(output->elemType());

    //Declare alpha and beta
    codeStream << indent(1) << "const half alpha = __float2half(1.0f);" << std::endl
               << indent(1) << "const half beta = __float2half(0.0f);" << std::endl;
    
    std::string M = genNumElem(output, 0, output->dimSizes().size() - 1);
    std::string N = genNumElem(output, output->dimSizes().size() - 1, output->dimSizes().size());
    std::string K = genNumElem(matmul->operand(0), matmul->operand(0)->dimSizes().size() - 1, matmul->operand(0)->dimSizes().size());
    std::stringstream cublasCall;
    //Always perform row major
    cublasCall << "cublasGemmEx(" << cublasHandleVar << ", CUBLAS_OP_N, CUBLAS_OP_N" << ", " << std::endl
               << indent(2) << N << ", " <<  M << ", " << K << ", " << std::endl
               << indent(2) << "&alpha, "
               << indent(2) << input2Name << ", " << cublasTypeA << ", " << N << ", " << std::endl
               << indent(2) << input1Name << ", " << cublasTypeB << ", " << K << ", " << std::endl
               << indent(2) << "&beta, " << name << ", " << cublasTypeC << ", " << N << ", " << std::endl
               << indent(2) << "CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP)";
    //TODO: Supports only 16 bit for now
    codeStream << indent(1) << cublasCheck(cublasCall.str()) << std::endl;
    codeStream << "}";
    return CFunc({funcName, codeStream.str(), inputs, true, MatMulNode});
}

CFunc generateNormCUDA(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<NormImpl> reduceTensor)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string funcName = "redOpFunc" + std::to_string(nameCounter++);    

    codeStream << "__global__ void " << funcName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    std::set<std::shared_ptr<ExpressionImpl>> redOpInputs;
    redOpInputs.insert(reduceTensor->arg());
    redOpInputs.insert(output);

    for (auto it : pipeline.explicitStoreLocations()) {
        redOpInputs.erase(it.first);
        redOpInputs.insert(it.second);
    }

    int ii = 0;
    for (auto iter : redOpInputs) {
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        if (ii != redOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }


    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx

    codeStream << iteratorInit(reduceTensor->arg()->dims(), indent(1)) << std::endl;
    
    //Print assignment to output stage
    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : pipeline.explicitStoreLocations().at(output)->name();
    std::string inputName;
    if (reduceTensor->arg()->type() == TensorNode || reduceTensor->arg()->type() == VariableNode) {
        inputName = reduceTensor->arg()->name();
    } else {
        std::shared_ptr<StageImpl> inputStage = AstNodeImpl::asStageImpl(reduceTensor->arg());
        inputName = pipeline.explicitStoreLocations().count(inputStage) == 0 ? reduceTensor->arg()->name() : pipeline.explicitStoreLocations().at(inputStage)->name();
    }

    codeStream << indent(1);
    //TODO: Improve this by using NVIDIA CUB, maybe?
    codeStream << "__atomicAdd(" << name << ", " << inputName << "[" << iteratorAccessString(output) << "] * " << inputName << "[" << iteratorAccessString(output) << "]" << ");" << std::endl;
    codeStream << "}";

    return CFunc({funcName, codeStream.str(), redOpInputs, true, NormNode});
}


CFunc generateDropoutCUDA(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<DropoutImpl> dropoutNode)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string binOpFuncName = "dropout" + std::to_string(nameCounter++);

    std::set<std::shared_ptr<ExpressionImpl>> binOpInputs = dropoutNode->usedExprs();

    codeStream << "__global__ void " << binOpFuncName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    binOpInputs.insert(output);

    for (auto it : pipeline.explicitStoreLocations()) {
        binOpInputs.erase(it.first);
        binOpInputs.insert(it.second);
    }
    auto dimExprs = allDimExprs(binOpInputs.begin(), binOpInputs.end());
    binOpInputs.insert(dimExprs.begin(), dimExprs.end());
    
    int ii = 0;
    for (auto iter : binOpInputs) {
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        if (ii != binOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }

    codeStream << ", " << commSizeTy << " " << commSizeArg << ", " << rankVarTy << " " << rankVar;
    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx

    //All code generated for binary operations will be single dimension.
    std::string varIterator = iteratorInit(1, indent(1));

    codeStream << varIterator << std::endl;

    //Print Binary operation
    std::stringstream binopCodeStream;
    PointwiseOpCodegen binOpCodegen(binopCodeStream, pipeline, false, CodeType::CUDA, isMixedPrecision(dropoutNode));
    binOpCodegen.print(*dropoutNode);
    
    //Add declarations
    codeStream << indent(1) << binOpCodegen.decls() << std::endl;

    //Print assignment to output stage
    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : pipeline.explicitStoreLocations().at(output)->name();
    codeStream << indent(1) << name;
    codeStream << "[" << iteratorForDim(0) << "]" 
               << " = " << binopCodeStream.str() << ";" << std::endl;

    codeStream << "}";

    return CFunc({binOpFuncName, codeStream.str(), binOpInputs, true, DropoutNode});
}

CFunc generateBinOpCodeCUDA(Pipeline& pipeline, std::shared_ptr<StageImpl> output, std::shared_ptr<BinaryPointwiseOp> binOpNode)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string binOpFuncName = "binOpFunc" + std::to_string(nameCounter++);

    std::set<std::shared_ptr<ExpressionImpl>> binOpInputs = binOpNode->usedExprs();

    codeStream << "__global__ void " << binOpFuncName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    binOpInputs.insert(output);

    for (auto it : pipeline.explicitStoreLocations()) {
        binOpInputs.erase(it.first);
        binOpInputs.insert(it.second);
    }
    auto dimExprs = allDimExprs(binOpInputs.begin(), binOpInputs.end());
    binOpInputs.insert(dimExprs.begin(), dimExprs.end());
    
    int ii = 0;
    for (auto iter : binOpInputs) {
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        if (ii != binOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }

    codeStream << ", " << commSizeTy << " " << commSizeArg << ", " << rankVarTy << " " << rankVar;
    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx

    //All code generated for binary operations will be single dimension.
    std::string varIterator = iteratorInit(1, indent(1));

    codeStream << varIterator << std::endl;

    //Print Binary operation
    std::stringstream binopCodeStream;
    PointwiseOpCodegen binOpCodegen(binopCodeStream, pipeline, false, CodeType::CUDA, isMixedPrecision(binOpNode));
    binOpCodegen.print(*binOpNode);
    
    //Print assignment to output stage
    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : pipeline.explicitStoreLocations().at(output)->name();
    codeStream << indent(1) << name;
    codeStream << "[" << iteratorForDim(0) << "]" 
               << " = " << binopCodeStream.str() << ";" << std::endl;

    codeStream << "}";

    return CFunc({binOpFuncName, codeStream.str(), binOpInputs, true, BinaryPointwiseOpNode});
}

CFunc generateFusedBinOpCodeCUDA(Pipeline& pipeline, PipelineStage* pipeStage)
{
    std::stringstream codeStream;
    static int nameCounter = 0;
    std::string binOpFuncName = "binOpFunc" + std::to_string(nameCounter++);
    InputsVisitor inputsVisitor;
    std::vector<std::shared_ptr<StageImpl>> outStages = pipeStage->stages();
    
    std::vector<std::shared_ptr<StageImpl>> normStages;
    std::set<std::shared_ptr<StageImpl>> intermediates;

    std::set<std::shared_ptr<ExpressionImpl>> binOpInputs;
    for (auto stage : pipeStage->stages()) {
        auto inputs = stage->usedExprs();
        binOpInputs.insert(inputs.begin(), inputs.end());
    }
    
    codeStream << "__global__ void " << binOpFuncName << "(";

    //Print arguments of CUDA Kernel
    //Add output to arguments too.
    auto liveouts = pipeStage->liveoutStages(pipeline.outputs());
    
    binOpInputs.insert(liveouts.begin(), liveouts.end());
    auto dimExprs = allDimExprs(binOpInputs.begin(), binOpInputs.end());
    binOpInputs.insert(dimExprs.begin(), dimExprs.end());

    //Remove all storeAt's target and add source
    for (auto it : pipeline.explicitStoreLocations()) {
        binOpInputs.erase(it.first);
        binOpInputs.insert(it.second);
    }

    //Add output of Norm as an input
    for (auto stage : pipeStage->stages()) {
        if (stage->definition()->type() == NormNode) {
            binOpInputs.insert(stage);
            normStages.push_back(stage);
            //Each norm requires a memory location and hence is an intermediate
            intermediates.insert(stage);
        }
    }

    //For each norm obtain all the stages that the norm depends on.
    //Also, find all stages that uses norm and a stage norm depends on.
    std::set<std::shared_ptr<StageImpl>> commonExprsForNorm;
    //TODO: need a better name 
    for (auto normStage : normStages) {
        auto dependsOnStages = normStage->dependsOnStages();
    
        for (auto stage : outStages) {
            auto usedExprs = stage->usedExprs();
            if (usedExprs.find(normStage) != usedExprs.end()) {
                auto intersection = setIntersection(dependsOnStages, usedExprs);
                commonExprsForNorm.insert(intersection.begin(), intersection.end());
            }
        }
    }

    for (auto commonExpr : commonExprsForNorm) {
        //All common expr are stored in memory
        pipeStage->setStorageLocation(commonExpr, Memory);
        intermediates.insert(commonExpr);
        binOpInputs.insert(commonExpr);
    }

    std::set<std::shared_ptr<ExpressionImpl>> arguments;

    int ii = 0;
    for (auto iter : binOpInputs) {
        if (iter->type() == StageNode && 
            pipeStage->getStorageLocation(AstNodeImpl::asStageImpl(iter)) == Register) {
            continue;
        }
        arguments.insert(iter);
        codeStream << elemTypeToCType(iter->elemType()) << " ";
        if (iter->type() == TensorNode || iter->type() == StageNode)
            codeStream << "* ";
        codeStream << iter->name();
        if (ii != binOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }

    codeStream << commSizeTy << " " << commSizeArg << ", " << rankVarTy << " " << rankVar;
    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx
    
    /**Generate all operations**/
    std::stringstream normInits;

    for (size_t it = 0; it < outStages.size(); it++) {
        std::stringstream binopCodeStream;
        std::shared_ptr<StageImpl> output = outStages[it];
        auto stageDef = output->definition();

        //Initialize Norm memory locs
        if (stageDef->type() == NormNode) {
            normInits << indent(2) << "*" << output->name() << " = 0;" << std::endl;
        }
    }

    int indentLevel = 1;

    if (normStages.empty()) {
        //Generate single dimension code for binary pointwise 
        codeStream << iteratorInit(1, indent(indentLevel)) << std::endl;
        // No norm stage, so it is fine.
        for (size_t it = 0; it < outStages.size(); it++) {
            std::stringstream binopCodeStream;
            std::shared_ptr<StageImpl> output = outStages[it];
            auto stageDef = output->definition();
            if (stageDef->type() == UpdateNode)
                stageDef = AstNodeImpl::asUpdateImpl(stageDef)->update();
            else {    
                std::shared_ptr<BinaryPointwiseOp> binOpNode = AstNodeImpl::asBinaryPointwiseOp(stageDef);
                PointwiseOpCodegen binOpCodegen(binopCodeStream, 
                                            pipeStage, pipeline, false, false, CodeType::CUDA, isMixedPrecision(binOpNode));
                binOpCodegen.print(*binOpNode);
                //Print assignment to output stage
                //If stored in a register then emit declaration
                if (pipeStage->getStorageLocation(output) == Register) {
                    codeStream << indent(indentLevel) << elemTypeToCType(output->elemType()) << " " << output->name() << ";" << std::endl;
                }
                std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : 
                                pipeline.explicitStoreLocations().at(output)->name();
                codeStream << indent(indentLevel) << name;
                if (pipeStage->getStorageLocation(output) == Memory) {
                    codeStream << "[" << iteratorForDim(0) << "]" ;
                }
                
                codeStream << " = " << binopCodeStream.str() << ";" << std::endl;
            }
        }
    } else {
        /* Norm fused with other operations is implemented using Cooperative Grid Group and grid strided loops
        * After computing the grid strided loop, a this_grid().sync() is generated.
        * Expressions using a norm output might also use a stage which has a RAW dependency with norm (i.e., a path from stage to the norm in the DAG).
        * These input stages cannot be stored in registers because they are in next grid strided loop.
        * Therefore, a norm "breaks" the fused loop and the input to norm is an intermediate.
        */

        codeStream << indent(indentLevel) << "if (" << threadIdxInGrid << " == 0) {" << std::endl;
        codeStream << normInits.str();
        codeStream << indent(indentLevel) << "}" << std::endl;
        codeStream << indent(indentLevel) << "this_grid().sync();" << std::endl;

        int it = 0;

        while (it < outStages.size()) {
            //Generate grid strided loops
            codeStream << indent(indentLevel) << "for (" << iteratorInit(1, indent(0)) << " " << iteratorForDim(0) << " < " << genNumElem(*liveouts.begin()) << "; " << iteratorForDim(0) << " += gridDim.x * blockDim.x) {" << std::endl;
            indentLevel += 1;

            for (; it < outStages.size() && outStages[it]->definition()->type() != NormNode; it++) {
                std::stringstream binopCodeStream;
                auto output = outStages[it];
                auto stageDef = output->definition();
                if (stageDef->type() == UpdateNode) {
                    stageDef = AstNodeImpl::asUpdateImpl(stageDef)->update();
                } else if (stageDef->type() == BinaryPointwiseOpNode) {    
                    std::shared_ptr<BinaryPointwiseOp> binOpNode = AstNodeImpl::asBinaryPointwiseOp(stageDef);
                    PointwiseOpCodegen binOpCodegen(binopCodeStream, 
                                                pipeStage, pipeline, false, false, CodeType::CUDA, isMixedPrecision(binOpNode));
                    binOpCodegen.print(*binOpNode);
                    //Print assignment to output stage
                    //If stored in a register then emit declaration
                    if (pipeStage->getStorageLocation(output) == Register) {
                        codeStream << indent(indentLevel) << elemTypeToCType(output->elemType()) << " " << output->name() << ";" << std::endl;
                    }
                    std::string name = pipeline.explicitStoreLocations().count(output) == 0 ? output->name() : 
                                    pipeline.explicitStoreLocations().at(output)->name();
                    codeStream << indent(indentLevel) << name;
                    if (pipeStage->getStorageLocation(output) == Memory) {
                        codeStream << "[" << iteratorForDim(0) << "]" ;
                    }
                    
                    codeStream << " = " << binopCodeStream.str() << ";" << std::endl;
                }
            }

            if (it < outStages.size()) {
                auto output = outStages[it];
                auto stageDef = output->definition();
                if (stageDef->type() == NormNode) {
                    auto norm = AstNodeImpl::asNormImpl(stageDef);
                    codeStream << indent(indentLevel) << "atomicAdd(" << output->name() << ", " << norm->arg()->name() <<"[" << iteratorForDim(0) << "]);" << std::endl;
                    //For Norm add atomic update and synchronize all thread blocks
                    indentLevel -= 1;
                    codeStream << indent(indentLevel) << "}" << std::endl;
                    codeStream << indent(indentLevel) << "this_grid().sync();" << std::endl;
                } else {
                    indentLevel -= 1;
                    codeStream << indent(indentLevel) << "}" << std::endl;
                }

                it++;
            } else {
                indentLevel -= 1;
                codeStream << indent(indentLevel) << "}" << std::endl;
            }
        }
    }

    codeStream << "}";

    return CFunc({binOpFuncName, codeStream.str(), arguments, true, BinaryPointwiseOpNode,
    !normStages.empty(), intermediates});
}

std::string generateStageCompUsingMULTI(Pipeline& pipeline, std::shared_ptr<StageImpl> stage, std::shared_ptr<StageImpl> commCollStage, std::string funcName,
                                        std::string commCollArgSubstitute = "") {
    std::stringstream primsComputation;
    primsComputation << indent(2) << "uint64_t " << stage->name() << " = MULTI<" << funcName << "<T>" << ", T>()." << funcName << "(";
            
    //FIXME: To the call to MULTI, we add all arguments other than 
    //the result of Stage and the argument of AllReduce.
    InputsVisitor inVisitor;
    auto inputs = inVisitor.inputs(*stage);

    for (auto iter = inputs.begin(); iter != inputs.end();) {
        auto arg = *iter;
        if (commCollArgSubstitute != "" and commCollStage.get() == arg)
            primsComputation << commCollArgSubstitute; 
        else {
            //If it is an argument to pipe then we emit "+ offset" too.
            //We do not check for explicit Store locations right now to transfer data through register.
            if (arg->type() == TensorNode) {
                primsComputation << "*(" << arg->name() << "Pack + offset)";
            } else {
                primsComputation << arg->name();
            }
        }

        if (++iter != inputs.end())
            primsComputation << ", ";
    }

    primsComputation << ");" << std::endl;
    return primsComputation.str();
}

template<class T>
std::string funcBodyForFusedBinOpCommCollCodeForNCCL(Pipeline& pipeline, PipelineStage* pipeStage, std::shared_ptr<StageImpl> output, 
                                                     std::string type, T& binOpInputs, std::vector<std::shared_ptr<CastImpl>>& mixedPrecisionCasts) 
{
    // if (output->definition()->type() != BinaryPointwiseOpNode) {
    //     return "";
    // }

    std::stringstream codeStream;
    std::vector<std::shared_ptr<StageImpl>> outStages = pipeStage->stages();

    codeStream << "\n  __device__ ";
    //Generate output stage type in case of mixed precision otherwise given type
    if (mixedPrecisionCasts.size() > 0) 
        codeStream << elemTypeToCType(output->elemType());
    else
        codeStream << type;
    codeStream << " operator()"<< "(";

    int ii = 0;
    for (auto iter : binOpInputs) {
        codeStream << "const ";
        //If this expr (i.e. iter) is input to cast expression then generate specified type
        //otherwise generate the type of expression.
        if (mixedPrecisionCasts.size () > 0) {
            bool inputToCast = false;

            for (auto cast : mixedPrecisionCasts) {
                if (cast->op() == iter) {
                    inputToCast = true;
                }
            }

            if (inputToCast) {
                codeStream << type << " ";    
            } else {
                codeStream << elemTypeToCType(iter->elemType()) << " ";
            }
        } else {
            codeStream << type << " ";
        }
        codeStream << iter->name();
        if (ii != binOpInputs.size() - 1)
            codeStream << ", ";
        ii++;
    }


    codeStream << ") {" << std::endl;

    //Function body
    //Iterator initialization from threadIdx and blockIdx
    
    //Print All Binary Operations
    //TODO: Need to generate one function for each Binary Operation
    std::stringstream binopCodeStream;
    // BinaryPointwiseOp* binOpNode = dynamic_cast<BinaryPointwiseOp*>(output->definition().get());
    
    //If there is a cast operation from a smaller bitwidth Stage 
    //to a larger bitwidth Stage then we consider that as a mixed precision.
    
    PointwiseOpCodegen binOpCodegen(binopCodeStream, 
                                          pipeStage, pipeline, false, true, CodeType::CUDA, 
                                          mixedPrecisionCasts);
    if (type == "half" || type == "half2") {
        binOpCodegen.setExplicitType(TensorElemType::Float16);
        if (type == "half2") binOpCodegen.setHalf2Type();
    }

    binOpCodegen.print(*output->definition());
    
    //Print assignment to output stage
    //If stored in a register then emit declaration
    // if (pipeStage->getStorageLocation(output) == Register) {
    //     codeStream << indent(1) << elemTypeToCType(output->elemType()) << " " << output->name() << ";" << std::endl;
    // }
    codeStream << indent(1) << " return " << binopCodeStream.str() << ";" << std::endl;

    codeStream << "}";

    return codeStream.str();
}
CStruct generateFusedBinOpCommCollCodeForNCCL(std::string name, Pipeline& pipeline, PipelineStage* pipeStage, 
                                              std::shared_ptr<StageImpl> outStage, bool useHalfType, std::vector<std::shared_ptr<CastImpl>>& mixedPrecisionCasts)
{
    std::string reduceKernelFuncTemplate = "template<typename T>"
                                                 "    struct <NAME> {"
                                                 "    <OPERATOR>"
                                                 "};";
    std::string binOpStructName = name;
    if (useHalfType) {
        reduceKernelFuncTemplate = reduceKernelFuncTemplate.replace(reduceKernelFuncTemplate.find("<typename T>"),
                                                                    std::string("<typename T>").size(), "<>");
    } 
    
    std::vector<std::shared_ptr<StageImpl>> outStages = pipeStage->stages();
    
    InputsVisitor inVisitor;
    ASSERT(false, "FIX");
    std::set<std::shared_ptr<ExpressionImpl>> binOpInputs;//FIX = inVisitor.inputs(*outStage);

    //Print arguments of CUDA Kernel
    //Add output to arguments too.

    // //Remove all storeAt's target and add source
    //FIX
    // for (auto it : pipeline.explicitStoreLocations()) {
    //     if (binOpInputs.find(it.first) != binOpInputs.end()) {
    //         binOpInputs.erase(it.first);
    //         binOpInputs.insert(it.second);
    //     }
    // }

    std::string funcCode = funcBodyForFusedBinOpCommCollCodeForNCCL(pipeline, pipeStage, outStage, useHalfType ? "half" : "T",
                                                                    binOpInputs, mixedPrecisionCasts);
    if (useHalfType) {
        std::string funcCode2 = funcBodyForFusedBinOpCommCollCodeForNCCL(pipeline, pipeStage, outStage, "half2",
                                                                         binOpInputs, mixedPrecisionCasts);
        funcCode = funcCode + "\n" + indent(1) + funcCode2;
    }

    std::string structBody = reduceKernelFuncTemplate;
    if (useHalfType) {
        std::string t = "half ";
        structBody = structBody.replace(structBody.find("<NAME>"), 
                                        std::string("<NAME>").size(), binOpStructName+"<"+t+">");
    } else {
        structBody = structBody.replace(structBody.find("<NAME>"), 
                                        std::string("<NAME>").size(), binOpStructName);
    }

    structBody = structBody.replace(structBody.find("<OPERATOR>"), 
                                    std::string("<OPERATOR>").size(), funcCode);
    std::cout << __LINE__ << " binOpStructName " << binOpStructName << std::endl;
    return CStruct({binOpStructName, structBody, true});
}

std::map<std::shared_ptr<StageImpl>, std::tuple<CStruct, CStruct>> generateFusedBinOpCommCollCodeForNCCL(Pipeline& pipeline, PipelineStage* pipeStage, 
                                                                                         std::map<std::shared_ptr<StageImpl>, std::vector<std::shared_ptr<CastImpl>>>& allMixedPrecisionCasts)
{
    static int nameCounter = 0;
    std::map<std::shared_ptr<StageImpl>, std::tuple<CStruct, CStruct>> compStructForOutput;

    for (auto output : pipeStage->stages()) {
        if (output->definition()->type() != BinaryPointwiseOpNode &&
            output->definition()->type() != UnaryPointwiseOpNode &&
            output->definition()->type() != IteNode) {
            continue;
        }
        ASSERT(false, "FIXME");
        #if 0
        std::string binOpFuncName = std::string((allMixedPrecisionCasts[output].size() > 0) ? "mixed" : "") + "binOp" + std::to_string(++nameCounter);
        CStruct genericStruct = generateFusedBinOpCommCollCodeForNCCL(binOpFuncName, pipeline, pipeStage, output, false, allMixedPrecisionCasts[output]);
        //Generate only generic struct if have mixedprecision casts
        if (allMixedPrecisionCasts[output].size() == 0) {
            CStruct halfStruct = generateFusedBinOpCommCollCodeForNCCL(binOpFuncName, pipeline, pipeStage, output, true, allMixedPrecisionCasts[output]);
            compStructForOutput[output] = std::make_tuple(genericStruct, halfStruct);
        } else {
            compStructForOutput[output] = std::make_tuple(genericStruct, genericStruct);
        }
        #endif
    }

    return compStructForOutput;
}

template<class T>
std::string MULTIMethodBodyFor4BytesType(std::shared_ptr<StageImpl> out, T& inputs, PipelineStage* pipelineStage, std::string type,
                                         const std::vector<std::string> members, bool useHalf2)
{
    std::stringstream body;

    for (auto member : members) {
        body << "c" << out->name() << "." << member << " = FUNC()(";
        for (auto it = inputs.begin(); it != inputs.end();) {
            if ((*it)->type() == VariableNode || 
                ((*it)->type() == StageNode && AstNodeImpl::asStageImpl(*it)->definition()->type() == ReduceTensorNode)) {
                if(type == "half" && useHalf2) {
                    body << "__half2half2(" << (*it)->name() << ")";
                } else {
                    body << (*it)->name();
                }
            } else {
                body << "c" << (*it)->name() << "." << member;
            }
            
            if (++it != inputs.end()) {
                body << ", ";
            }
        }

        body << ");" << std::endl;
    }

    return body.str();
}

template<class T>
std::string MULTIMethodBodyFor4BytesTypeMixedPrecision(std::shared_ptr<StageImpl> out, T& inputs, PipelineStage* pipelineStage, std::string type,
                                                       const std::vector<std::string> members, bool useHalf2, std::string packStructName)
{
    std::stringstream body;

    for (auto member : members) {
        body << "c" << out->name() << "." << packStructName << "." << member << " = FUNC()(";
        for (auto it = inputs.begin(); it != inputs.end();) {
            if ((*it)->type() == VariableNode) {
                if(type == "half" && useHalf2) {
                    body << "__half2half2(" << (*it)->name() << ")";
                } else {
                    body << (*it)->name();
                }
            } else {
                body << "c" << (*it)->name() << "." << (((*it)->elemType() != Float16) ? packStructName + ".": "") << "get" << member << "()";
            }
            
            if (++it != inputs.end()) {
                body << ", ";
            }
        }

        body << ");" << std::endl;
    }

    return body.str();
}

std::string generateFusedNCCLCommColl(Pipeline& pipeline, PipelineStage* pipelineStage)
{
    const std::string ACCCDSL_PATH = "../../";
    const std::string NCCL_SRC_PATH = ACCCDSL_PATH + "/nccl/";
    const std::string NCCL_DST_PATH = ACCCDSL_PATH + "../nccl/";
    const std::string INSERT_TAG = "/*{INSERT HERE}*/";
    const std::string INSERT_SIZE_TAG = "/*{INSERT SIZE HERE}*/";
    const std::string ncclFuncResultTy = "ncclResult_t";
    /*Other Arguments for nccl functions*/
    const std::string sizeArg = "count";
    const std::string sizeArgTy = "size_t";
    const std::string ncclTypeArg = "datatype";
    const std::string ncclTypeArgTy = "ncclDataType_t";
    const std::string ncclCommArg = "comm";
    const std::string ncclCommArgTy = "ncclComm_t";
    const std::string streamArg = "stream";
    const std::string streamArgTy = "cudaStream_t";
    const std::string redOp = "op";
    const std::string redOpTy = "ncclRedOp_t";
    const std::string BEGIN_READ_TAG = "/*{BEGIN-READ}*/";
    const std::string END_READ_TAG = "/*{END-READ}*/";
    const std::string CC = "gcc";

    AstNodeType commCollType;
    std::set<std::shared_ptr<StageImpl>> commCollStages;
    ReduceOperation commCollRedOp;
    std::shared_ptr<ExpressionImpl> commCollArg;
    std::shared_ptr<StageImpl> commCollOutput;

    for (auto outStage : pipelineStage->stages()) {
        std::shared_ptr<ExpressionImpl> stageDef = outStage->definition();
        if (stageDef->isCommCollective()) {
            commCollType = outStage->definition()->type();
            commCollRedOp = getCommCollRedOp(*outStage->definition().get());
            commCollStages.insert(outStage);
            commCollOutput = outStage;
            switch (commCollType) {
                case AllReduceNode:
                    commCollArg = AstNodeImpl::asAllReduceImpl(outStage->definition())->arg();
                    break;
            }
        }
    }

    if (pipelineStage->getFusedIntoCollComm() != NoneCollCommOp) {
        commCollType = (pipelineStage->getFusedIntoCollComm() == AllReduceOp) ? AllReduceNode : AllGatherNode;
        
        std::vector<AstNodeType> commCollTypes;
        if (commCollStages.size() == 2) {
            for (auto stage : commCollStages) {
                commCollTypes.push_back(stage->definition()->type());
            }
        }

        if (commCollType == AllReduceNode) {
            //If AllReduce is fused with ReduceScatter and AllGather then commCollArg is arg of ReduceScatter
            ASSERT((commCollTypes[0] == ReduceScatterNode && commCollTypes[1] == AllGatherNode) || 
                   (commCollTypes[1] == ReduceScatterNode && commCollTypes[0] == AllGatherNode),
                   "Only fusion of ReduceScatter and AllGather into AllReduce is supported\n");
            std::shared_ptr<StageImpl> reduceScatterStage = nullptr;
            for (auto stage: commCollStages) {
                if (stage->definition()->type() == ReduceScatterNode) {
                    reduceScatterStage = stage;
                    commCollOutput = stage;
                    break;
                }
            }

            commCollArg = AstNodeImpl::asReduceScatterImpl(commCollOutput->definition())->arg();
        }
    } 
    
    std::stringstream ncclFuncArgs;
    std::string funcName = "AllReduce_pipe";
    //  AstNodeTypeToStr(commCollType);
    // funcName = funcName.substr(0, funcName.find("Node")) + "_" + pipeline.name();
    std::set<std::shared_ptr<ExpressionImpl>> gpuKernelArgs;
    std::set<std::shared_ptr<ExpressionImpl>> outputs;
    std::vector<std::shared_ptr<StageImpl>> computationStages;
    std::set<std::shared_ptr<ExpressionImpl>> outputStages;

    for (auto stage : pipelineStage->stages()) {
        if (commCollStages.count(stage) == 0 and 
            stage->definition()->type() != ReduceTensorNode)
            computationStages.push_back(stage);
    }

    for (auto arg : pipelineStage->liveinExprs()) {
        gpuKernelArgs.insert(arg);
    }

    for (auto out : pipelineStage->liveoutStages(pipeline.outputs())) {
        outputStages.insert(out);
        if (pipeline.explicitStoreLocations().count(out) > 0) {
            outputs.insert(pipeline.explicitStoreLocations().at(out));
            gpuKernelArgs.insert(pipeline.explicitStoreLocations().at(out));
        }
        else {
            outputs.insert(out);
            gpuKernelArgs.insert(out);
        }
    }

    // ASSERT(outputStages.size() == 1, "Codegen for only one live out is supported but live outs are " << outputStages.size());

    for (auto arg : gpuKernelArgs) {
        ncclFuncArgs << printArgument(arg) << ", ";
    }

    ncclFuncArgs << sizeArgTy << " " << sizeArg << ", "
                 << ncclTypeArgTy << " " << ncclTypeArg << ", "
                 << ncclCommArgTy << " " << ncclCommArg << ", "
                 << ((commCollType == AllReduceNode || commCollType == ReduceNode || commCollType == ReduceScatterNode) ?
                     redOpTy +" " + redOp + ", ": "")
                 << streamArgTy << " " << streamArg;

    //Add declaration to nccl.h.in
    std::string ncclFunc = ncclFuncResultTy + " "  + funcName + "(" +
                               ncclFuncArgs.str() + ")";
    std::string ncclFuncDecl = ncclFunc +";\n";

    if (false) {
        std::string ncclHContents = readFile(NCCL_SRC_PATH + "nccl.h.in");
        ncclHContents = ncclHContents.replace(ncclHContents.find(INSERT_TAG), 
                                              INSERT_TAG.size(), ncclFuncDecl);
        writeFile(NCCL_DST_PATH + "nccl.h.in", ncclHContents);
    }

    /*Create the function call*/
    std::stringstream ncclFuncCall;

    ncclFuncCall << funcName << "(";

    for (auto arg : gpuKernelArgs) {
        ncclFuncCall << arg->name() << ", ";
    }

    ncclFuncCall << genNumElem(*outputs.begin()) << ", " << elemTypeToNCCLType((*commCollStages.begin())->elemType())
                 << ", " << ncclCommArg << ", " << 
                    (commCollRedOp != ReduceOperationNone ?
                     redOpToNCCLReduceOp(commCollRedOp) + ", ": "")
                 << streamArg << ")";

    std::string ncclFuncCallStr = ncclCheck(ncclFuncCall.str());
    
    std::string v = "";
    
    if (pipeline.name() == "adam") {
        v = "0";
        ncclFuncCallStr = "NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, (half*)g, w, (half*)w, m, v, N, ncclHalf, comm, ncclSum, stream));";
    }
    else if (pipeline.name() == "lamb") {
        v = "1";
        ncclFuncCallStr = "NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, (half*)g, w, (half*)w, m, v, w, N, ncclHalf, comm, ncclSum, stream));";
    } else {
        v = "0";
        ncclFuncCallStr = "NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, (half*)g, w, (half*)w, m, v, N, ncclHalf, comm, ncclSum, stream));";
    }

    std::cout << "v " << v << " " << pipeline.name() << std::endl;
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/nccl.h.in", 
                              "#define TYPE_NCCL_H_IN\\s*\\d*", "#define TYPE_NCCL_H_IN " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/common_kernel.h", 
                              "#define TYPE_COMMON_KERNEL\\s*\\d*", "#define TYPE_COMMON_KERNEL " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/prims_ll128_computation.h", 
                              "#define TYPE_PRIMS_LL128\\s*\\d*", "#define TYPE_PRIMS_LL128 " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/primitives_computation.h", 
                              "#define TYPE_PRIMS\\s*\\d*", "#define TYPE_PRIMS " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/prims_ll_computation.h", 
                              "#define TYPE_PRIMS_LL\\s*\\d*", "#define TYPE_PRIMS_LL " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/all_reduce_computation.h", 
                              "#define TYPE_ALL_REDUCE\\s*\\d*", "#define TYPE_ALL_REDUCE " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/reduce_kernel.h", 
                              "#define TYPE_REDUCE_KERNEL\\s*\\d*", "#define TYPE_REDUCE_KERNEL " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/device/all_reduce.h", 
                              "#define TYPE_ALL_REDUCE\\s*\\d*", "#define TYPE_ALL_REDUCE " + v+"\n");
    replaceAllSubStringInFile(NCCL_SRC_PATH+"/src/collectives/all_reduce.cc", 
                              "#define TYPE_ALL_REDUCE\\s*\\d*", "#define TYPE_ALL_REDUCE " + v+"\n");

    // ./collectives/device/prims_ll128_computation.h:#define TYPE_PRIMS_LL128 1
    // ./collectives/device/all_reduce_computation.h:#define TYPE_ALL_REDUCE 1
    // ./collectives/device/primitives_computation.h:#define TYPE_PRIMS 1
    // ./collectives/device/prims_ll_computation.h:#define TYPE_PRIMS_LL 1
    // ./collectives/device/reduce_kernel.h:#define TYPE_REDUCE_KERNEL 1
    // ./collectives/all_reduce.cc:#define TYPE_ALL_REDUCE 1

    return ncclFuncCallStr;

    /*Update ncclInfo_t in include/info.h.in*/
    
    //Add all inputs and outputs to ncclInfo_t
    std::stringstream ncclInfoAdditions;
    for (auto arg : gpuKernelArgs) {
        if (arg->type() == VariableNode && arg->elemType() == Float16) {
            ncclInfoAdditions << printDeclaration(arg, ";", "", false, false, Float32) << std::endl;
        } else {
            ncclInfoAdditions << printDeclaration(arg) << std::endl;
        }
    }

    {
        std::string info_H = readFile(NCCL_SRC_PATH + "include/info.h.in");
        info_H = info_H.replace(info_H.find(INSERT_TAG), 
                                INSERT_TAG.size(), ncclInfoAdditions.str());
        writeFile(NCCL_DST_PATH + "include/info.h", info_H);
    }
    
    /*Update struct CollectiveArgs in include/devcomm.h.in*/
    std::string devcommH = readFile(NCCL_SRC_PATH + "include/devcomm.h.in");
    const size_t beginReadTagPos = devcommH.find(BEGIN_READ_TAG) + BEGIN_READ_TAG.size();
    const size_t endReadTagPos = devcommH.find(END_READ_TAG);
    std::string structStr = devcommH.substr(beginReadTagPos, endReadTagPos - beginReadTagPos);    
    structStr = structStr.replace(structStr.find(INSERT_TAG), INSERT_TAG.size(), 
                                  ncclInfoAdditions.str());
    structStr = structStr.replace(structStr.find(INSERT_SIZE_TAG), INSERT_SIZE_TAG.size(), 
                                  std::to_string(1));

    //Write struct to a C file, compile it and get the sizeof(ncclColl)
    char cFileName[] = "/tmp/XXXXXX.c";
    int cFileFd = mkstemps(cFileName, 2);
    ASSERT(cFileFd != -1, "Error in mkstemps ");

    const std::string cFileString = "#include <stdio.h>\n #include <stdint.h>\n struct halfStruct {char x,y;};\n typedef struct halfStruct half;\n" + structStr + "\n int main() {printf(\"%ld\\n\", sizeof(struct ncclColl));}";
    writeFile(cFileFd, cFileString);

    char compiledFileName[] = "./XXXXXX";
    int compiledFileFd = mkstemp(compiledFileName);
    close(compiledFileFd);

    const std::string command = CC + " " + cFileName + " -o " + compiledFileName;
    const std::string result = exec(command);

    const std::string sizeOfncclCollStr = exec(compiledFileName);
    remove(compiledFileName);

    size_t sizeOfncclColl = atol(sizeOfncclCollStr.c_str());

    sizeOfncclColl = currOrNextPowerOf2(sizeOfncclColl);

    //Replace tags in devcomm.h
    devcommH = devcommH.replace(devcommH.find(INSERT_TAG), INSERT_TAG.size(), 
                                ncclInfoAdditions.str());
    devcommH = devcommH.replace(devcommH.find(INSERT_SIZE_TAG), INSERT_SIZE_TAG.size(),
                                std::to_string(sizeOfncclColl/sizeof(int)));
    writeFile(NCCL_DST_PATH + "include/devcomm.h", devcommH);

    /*Write ncclInfo to ncclColl copying code in enqueue.cc.in*/
    std::stringstream ncclEnqueueCC; 
    
    for (auto arg : gpuKernelArgs) {
        ncclEnqueueCC << "coll->args." << arg->name() << " = " 
                      << "info->fusedComputation." << arg->name() << ";" << std::endl;
    }

    std::string enqueueCC = readFile(NCCL_SRC_PATH+"enqueue.cc.in");

    enqueueCC = enqueueCC.replace(enqueueCC.find(INSERT_TAG), INSERT_TAG.size(),
                                  ncclEnqueueCC.str());
    writeFile(NCCL_DST_PATH+"enqueue.cc", enqueueCC);

    /*Add declaration and function body in collectives/allreduce.cc.in*/
    std::string ncclAPIText;
    ncclAPIText = "NCCL_API("+ncclFuncResultTy+", " + funcName + ", " + 
                  ncclFuncArgs.str() + ");\n";

    std::stringstream ncclFuncBody, fusedComputationStruct;
    
    switch (commCollType) {
        case AllReduceNode:
        {
            std::map<std::shared_ptr<StageImpl>, std::vector<std::shared_ptr<CastImpl>>> stageToMixedPrecisions;
            for (auto stage : pipelineStage->stages()) {
                if (stage->definition()->type() == BinaryPointwiseOpNode) {
                    auto allCasts = isMixedPrecision(AstNodeImpl::asBinaryPointwiseOp(stage->definition()));
                    if (allCasts.size() > 0)
                        stageToMixedPrecisions[stage] = allCasts;
                }
            }

            //TODO: Right now even if there is one mixed precision binary op, code moves whole 
            //computation in the allReduce<ALGO><PROTO>Kernel, instead of in primitives_computation.

            //Check if output of AllReduce is of lower precision than the result of binop it is used in.
            bool onlyCommCollStageIsMixedPrec = false;
            for (auto stage : pipelineStage->stages()) {
                if (stage->definition()->type() == BinaryPointwiseOpNode and 
                    stageToMixedPrecisions.count(stage) > 0) {
                    InputsVisitor inputVisitor;
                    auto inputs = inputVisitor.inputs(*AstNodeImpl::asBinaryPointwiseOp(stage->definition()).get());
                    if (sizeOfElemType((*commCollStages.begin())->elemType()) < sizeOfElemType(stage->elemType())) {
                        onlyCommCollStageIsMixedPrec = true;
                    } else {
                        onlyCommCollStageIsMixedPrec = false;
                    }
                }
            }

            bool fusedToAllReduce = pipelineStage->getFusedIntoCollComm() != NoneCollCommOp;
            std::shared_ptr<StageImpl> allGatherStage = nullptr;
            std::shared_ptr<StageImpl> reduceScatterStage = nullptr;
            std::vector<std::shared_ptr<StageImpl>> stagesAfterAllGather;

            if (fusedToAllReduce) {
                //If has AllGather then get that stage.
                for (auto stage : commCollStages) {
                    if (stage->definition()->type() == AllGatherNode) {
                        allGatherStage = stage;
                    }

                    if (stage->definition()->type() == ReduceScatterNode) {
                        reduceScatterStage = stage;
                    }
                }

                ASSERT(allGatherStage != nullptr, "Current fusion of ReduceScatter+AllGather is supported to AllReduce.");

                bool foundAllGather = false;

                for (auto stage : pipelineStage->stages()) {
                    if (stage->definition()->type() == AllGatherNode) {
                        foundAllGather = true;
                    }

                    if (foundAllGather) {
                        stagesAfterAllGather.push_back(stage);
                    }
                }
            }

            //TODO: Following code only works if commCollStage has lower precision.

            if (stageToMixedPrecisions.size () > 0) {
                ASSERT(onlyCommCollStageIsMixedPrec, "Mixed precision code generation is supported only"\
                                                     "when output of AllReduce is of mixed precision");
            }

            //Determine if ReduceTensor on the output of AllReduce is done.
            int numReduceStages = 0;
            for (auto stage : pipelineStage->stages()) {
                if (stage->definition()->type() == ReduceTensorNode) {
                    numReduceStages++;        
                }
            }

            if (numReduceStages > 0) {
                ASSERT(numReduceStages == 1, "At maximum only one ReduceTensor is allowed.");
            }

            bool hasReduceTensor = false;
            std::shared_ptr<StageImpl> reduceTensorStage = nullptr;
            for (auto stage : pipelineStage->stages()) {
                if (stage->definition()->type() == ReduceTensorNode) {
                    hasReduceTensor = true;
                    reduceTensorStage = stage;
                    break;
                }
            }

            std::vector<std::shared_ptr<StageImpl>> reduceTensorDAG;
            // This may overlap with reduceTensorDAG since some things may have to be recomputed
            std::vector<std::shared_ptr<StageImpl>> afterReduceTensorDAG;

            if (hasReduceTensor) {
                //TODO: Create a function for this.
                //Find the DAG with ReduceTensor stage as output

                //Do a backward BFS
                std::set<std::shared_ptr<StageImpl>> dag;
                std::queue<std::shared_ptr<StageImpl>> q;

                q.push(reduceTensorStage);
                while(q.size() > 0) {
                    std::shared_ptr<StageImpl> t = q.front();
                    q.pop();
                    dag.insert(t);
                    auto def = t->definition();
                    std::set<std::shared_ptr<ExpressionImpl>> inputs;
                    inputs = def->usedExprs();

                    //In finding all inputs, do not take CommCollStage input in account because it is fused.
                    for (auto in : inputs) {
                        if (in->type() == StageNode) {
                            q.push(AstNodeImpl::asStageImpl(in));
                        }
                    }
                }

                //add these in topological order.
                while (reduceTensorDAG.size() != dag.size()) {
                    for (auto stage : pipelineStage->stages()) {
                        if (std::find(dag.begin(), dag.end(), stage) != dag.end()) {
                            reduceTensorDAG.push_back(stage);
                        }
                    }
                }

                for (auto stage : pipelineStage->stages()) {
                    if (dag.find(stage) == dag.end()) {
                        q.push(stage);
                    }
                }
                std::set<std::shared_ptr<StageImpl>> dagAfter;
                while(q.size() > 0) {
                    auto t = q.front();
                    q.pop();

                    // TODO: also skip when (dag.find(t) != dag.end() && <stage is stored into memory>)
                    if (t == reduceTensorStage) {
                        continue;
                    }

                    dagAfter.insert(t);
                    InputsVisitor inputsVisitor;
                    std::set<std::shared_ptr<ExpressionImpl>> inputs;
                    ExpressionImpl* def = t->definition().get();
                    inputs = def->usedExprs();
                    //In finding all inputs, do not take CommCollStage input in account because it is fused.

                    for (auto in : inputs) {
                        if (in->type() == StageNode) {
                            q.push(AstNodeImpl::asStageImpl(in));
                        }
                    }
                }
                for (auto stage : pipelineStage->stages()) {
                    if (std::find(dagAfter.begin(), dagAfter.end(), stage) != dagAfter.end()) {
                        afterReduceTensorDAG.push_back(stage);
                    }
                }
            }


            //All Read, Writes, and Computation of pipeline output will be 
            //performed using uint64_t.
            //Get a type that contains same number of elements of allreduce output
            //as does a uint64_t contains of the pipeline's output.
            const size_t elemsIn64Bits = sizeof(uint64_t)/sizeOfElemType((*outputs.begin())->elemType());
            const size_t allReduceInputVecSize = elemsIn64Bits*sizeOfElemType((*commCollStages.begin())->elemType());
            const std::string vecTypeName = uintTypeForSize(allReduceInputVecSize);
            std::unordered_map<std::shared_ptr<StageImpl>, std::string> vecTypeForStage;

            for (auto stage : pipelineStage->stages()) {
                vecTypeForStage[stage] = uintTypeForSize(elemsIn64Bits*sizeOfElemType(stage->elemType()));
            }

            std::set<std::shared_ptr<ExpressionImpl>> argsOtherThanInputandOutput;
            for (auto arg : gpuKernelArgs) {
                if (outputs.count(arg) != 0) 
                    continue;
                
                if (commCollArg == arg)
                    continue;
                
                argsOtherThanInputandOutput.insert(arg);
            }
            
            fusedComputationStruct << "struct ncclFusedComputationParams fusedComputationParams = {";

            for (auto iter = gpuKernelArgs.begin(); iter != gpuKernelArgs.end();) {
                auto arg = *iter;
                if (arg->type() == VariableNode && arg->elemType() == Float16) {
                    fusedComputationStruct << "(float)" << arg->name();
                } else {
                    fusedComputationStruct << arg->name();
                }

                if (iter++ != gpuKernelArgs.end()) {
                    fusedComputationStruct << ", ";
                }
            }

            fusedComputationStruct << "};" << std::endl;
            ncclFuncBody << ncclAPIText << std::endl << ncclFunc << "{" << std::endl;
            ncclFuncBody << fusedComputationStruct.str() << "struct ncclInfo info = {";

            ncclFuncBody << "ncclCollAllReduceComputation, \"AllReduceComputation\", nullptr, nullptr, ";
            ncclFuncBody << sizeArg << ", " << ncclTypeArg << ", " << redOp << ", 0,"
                         << ncclCommArg <<", " << streamArg << ", "
                        << "ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, fusedComputationParams";
            ncclFuncBody << " };" << std::endl;
            ncclFuncBody << "  return ncclEnqueueCheck(&info);\n}";

            std::string allreduceContents = readFile(NCCL_SRC_PATH + "collectives/all_reduce.cc.in");
            allreduceContents = allreduceContents.replace(allreduceContents.find(INSERT_TAG), 
                                                          INSERT_TAG.size(), ncclFuncBody.str());
            writeFile(NCCL_DST_PATH + "collectives/all_reduce.cc", allreduceContents);

            /*Write to collectives/device/all_reduce_computation.h*/

            std::stringstream sendCall;
            sendCall << "send(";
            sendCall << commCollArg->name() << "+ offset";
            sendCall << ", ";
            sendCall << "nelem);";

            std::stringstream recvReduceSendCall;

            recvReduceSendCall << "recvReduceSend(";
            recvReduceSendCall << commCollArg->name() << "+ offset" << ", ";
            recvReduceSendCall << "nelem);";

            std::stringstream recvReduceCopySendCall;
            std::stringstream perGPUReductionComp;
            std::stringstream outputSendCall;
            if (onlyCommCollStageIsMixedPrec || hasReduceTensor) {
                std::string inputName = commCollArg->name();
                recvReduceCopySendCall << "recvReduceCopy(" << inputName << " + offset, " 
                                       << "(T*)(" << inputName << " + offset), ";  
                
                if (onlyCommCollStageIsMixedPrec) {
                    std::string outputCTypeStr = elemTypeToCType((*outputs.begin())->elemType());
                    outputSendCall << "send("<<"(T*)(((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << ") + offset), nelem * " << sizeOfElemType((*outputs.begin())->elemType())/sizeOfElemType((*commCollStages.begin())->elemType()) << ");";
                }
            } else {
                recvReduceCopySendCall << "recvReduceCopySend(";
                for (auto arg : gpuKernelArgs) {
                    if (arg->type() == VariableNode) {
                        recvReduceCopySendCall << arg->name();
                    } else {
                        recvReduceCopySendCall << arg->name() << " + offset";
                    }

                    recvReduceCopySendCall << ", ";
                }
            }

            std::stringstream recvCopySendCall;
            std::string recvCopySendName = "recvCopySend";
            if (fusedToAllReduce) {
                recvCopySendName += "AllGatherCompute";
            }
            if (onlyCommCollStageIsMixedPrec) {
                std::string outputCTypeStr = elemTypeToCType((*outputs.begin())->elemType());
                recvCopySendCall << recvCopySendName << "("<<"(T*)(((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << ") + offset), ";
            } else {
                recvCopySendCall << recvCopySendName << "(";
                for (auto arg : outputs) {
                    if (arg->type() == VariableNode) {
                        recvCopySendCall << arg->name();
                    } else {
                        recvCopySendCall << arg->name() << " + offset";
                    }

                    recvCopySendCall << ", ";
                }
            }

            std::stringstream recvCall;
            if (fusedToAllReduce) {
                recvCall << "recvAllGatherCompute(";
            } else {
                recvCall << "recv(";
            }
            if (onlyCommCollStageIsMixedPrec) {
                std::string outputCTypeStr = elemTypeToCType((*outputs.begin())->elemType());
                recvCall <<"(T*)(((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << ") + offset),";
            } else {
                for (auto arg : outputs) {
                    if (arg->type() == VariableNode) {
                        recvCall << arg->name();
                    } else {
                        recvCall << arg->name() << " + offset";
                    }

                    recvCall << ", ";
                }
            }

            /**********Writing to ncclAllReduceRingLLKernel**********/
            std::stringstream ringLLArgs;

            for (auto arg : gpuKernelArgs) {
                if (commCollArg == arg) {
                    ringLLArgs << indent(1) << "const " << printDeclarationForNCCLPrims(arg, true) << std::endl;
                    ringLLArgs << indent(1) << arg->name() << " = (const T" << ((arg->type() == VariableNode) ? ")" : "*)")
                               << "args->" << arg->name() << ";" << std::endl;
                } else {
                    ringLLArgs << indent(1) << printDeclarationForNCCLPrims(arg, true) << std::endl;
                    ringLLArgs << indent(1) << arg->name() << " = (T" << ((arg->type() == VariableNode) ? ")" : "*)")
                               << "args->" << arg->name() << ";" << std::endl;
                }
            }

            size_t sizeFactor = sizeOfElemType((*outputs.begin())->elemType())/sizeOfElemType((*commCollStages.begin())->elemType());
            std::string llprimsSend = "LLprims." + sendCall.str();
            std::string llprimsRecvReduceSend = "LLprims." + recvReduceSendCall.str();
            std::string llprimsRecvReduceCopySend = "LLprims." + recvReduceCopySendCall.str() + " nelem);";
            std::string llprimsRecvCopySend = "LLprims." + recvCopySendCall.str() + " nelem * " + std::to_string(sizeFactor) + ");";
            std::string llprimsRecv = "LLprims." + recvCall.str() + " nelem * " + std::to_string(sizeFactor) + ");";

            const std::string RINGLL_INSERT_ARGS_TAG = "/*RINGLL: {INSERT ARGS}*/";
            const std::string RINGLL_INSERT_SEND_TAG = "/*RINGLL: {INSERT SEND}*/";
            const std::string RINGLL_INSERT_RECV_REDUCE_SEND_TAG = "/*RINGLL: {INSERT RECV_REDUCE_SEND}*/";
            const std::string RINGLL_INSERT_RECV_REDUCE_COPY_SEND = "/*RINGLL: {INSERT RECV_REDUCE_COPY_SEND}*/";
            const std::string RINGLL_INSERT_RECV_COPY_SEND = "/*RINGLL: {INSERT RECV_COPY_SEND}*/";
            const std::string RINGLL_INSERT_RECV = "/*RINGLL: {INSERT RECV}*/";
            const std::string RINGLL_REDUCTION_END_LOOP = "/*RINGLL: REDUCTION {END FOR LOOP FOR}*/";
            const std::string RINGLL_REDUCTION_PER_GPU = "/*RINGLL: REDUCTION {PER-GPU REDUCTION}*/";
            const std::string RINGLL_REDUCTION_TRANSFER = "/*RINGLL: REDUCTION {TRANSFER}*/";
            const std::string RINGLL_REDUCTION_COMPUTATION = "/*RINGLL: REDUCTION {COMPUTATION}*/";
            const std::string RINGLL_REDUCTION_BEGIN_LOOP = "/*RINGLL: REDUCTION {BEGIN FOR LOOP FOR}*/";
            const std::string RINGLL_SHMEM_DECL_FOR_REDUCTION = "/*RINGLL: {INSERT SHARED MEMORY FOR REDUCTION}*/";

            const std::string RINGLL128_INSERT_ARGS = "/*RINGLL128: {INSERT ARGS}*/";
            const std::string RINGLL128_INSERT_SEND = "/*RINGLL128: {INSERT SEND}*/";
            const std::string RINGLL128_INSERT_RECV_REDUCE_SEND = "/*RINGLL128: {INSERT RECV_REDUCE_SEND}*/";
            const std::string RINGLL128_INSERT_RECV_REDUCE_COPY_SEND = "/*RINGLL128: {INSERT RECV_REDUCE_COPY_SEND}*/";
            const std::string RINGLL128_INSERT_RECV_COPY_SEND = "/*RINGLL128: {INSERT RECV_COPY_SEND}*/";
            const std::string RINGLL128_INSERT_RECV = "/*RINGLL128: {INSERT RECV}*/";

            std::string allReduceH = readFile(NCCL_SRC_PATH+"collectives/device/all_reduce_computation.h.in");
            allReduceH = replaceAllSubString(allReduceH, RINGLL_INSERT_ARGS_TAG, 
                                             ringLLArgs.str());
            allReduceH = replaceAllSubString(allReduceH, RINGLL128_INSERT_ARGS, 
                                             ringLLArgs.str());
            allReduceH = replaceAllSubString(allReduceH, RINGLL_INSERT_SEND_TAG, 
                                             llprimsSend);
            allReduceH = replaceAllSubString(allReduceH, RINGLL128_INSERT_SEND, 
                                             llprimsSend);
            allReduceH = replaceAllSubString(allReduceH, RINGLL_INSERT_RECV_REDUCE_SEND_TAG, 
                                             llprimsRecvReduceSend);
            allReduceH = replaceAllSubString(allReduceH, RINGLL128_INSERT_RECV_REDUCE_SEND, 
                                             llprimsRecvReduceSend);

            if (onlyCommCollStageIsMixedPrec) {
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_INSERT_RECV_REDUCE_COPY_SEND), 
                                                RINGLL_INSERT_RECV_REDUCE_COPY_SEND.size(), 
                                                llprimsRecvReduceCopySend + 
                                                "/*RINGLL: {INSERT MIXED-PRECISION COMPUTATION}*/\nLLprims." + outputSendCall.str());
                allReduceH = replaceAllSubString(allReduceH, RINGLL128_INSERT_RECV_REDUCE_COPY_SEND,
                                                llprimsRecvReduceCopySend + 
                                                "/*RINGLL128: {INSERT MIXED-PRECISION COMPUTATION}*/\nLLprims." + outputSendCall.str());
            }
            else {
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_INSERT_RECV_REDUCE_COPY_SEND), 
                                                RINGLL_INSERT_RECV_REDUCE_COPY_SEND.size(), llprimsRecvReduceCopySend);
                allReduceH = replaceAllSubString(allReduceH, RINGLL128_INSERT_RECV_REDUCE_COPY_SEND, 
                                                 llprimsRecvReduceCopySend);
            }
            allReduceH = allReduceH.replace(allReduceH.find(RINGLL_INSERT_RECV_COPY_SEND), 
                                            RINGLL_INSERT_RECV_COPY_SEND.size(), llprimsRecvCopySend);
            allReduceH = allReduceH.replace(allReduceH.find(RINGLL_INSERT_RECV), 
                                            RINGLL_INSERT_RECV.size(), llprimsRecv);

            allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_RECV_COPY_SEND), 
                                            RINGLL128_INSERT_RECV_COPY_SEND.size(), llprimsRecvCopySend);
            allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_RECV), 
                                            RINGLL128_INSERT_RECV.size(), llprimsRecv);
            allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_RECV_COPY_SEND), 
                                            RINGLL128_INSERT_RECV_COPY_SEND.size(), llprimsRecvCopySend);
            allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_RECV), 
                                            RINGLL128_INSERT_RECV.size(), llprimsRecv);
            writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceH);

            /*Add overloaded operator() for MULTI in collectives/device/common_kernel.h*/
            std::string INSERT_MULTI_OPERATOR = "/*{INSERT MULTI<T> operator()}*/";
            const std::string INSERT_REDUCE_KERNEL_HALF = "/*{INSERT REDUCE KERNEL<HALF> HERE}*/";
            const std::string INSERT_REDUCE_KERNEL_HALF2 = "/*{INSERT REDUCE KERNEL<HALF2> HERE}*/";

            const std::vector<std::string> ncclSupportedTypes {"uint8_t", "int8_t", "half", "uint32_t", "int32_t", 
                                                               "float", "uint64_t", "int64_t", "double"};
            //A map of computation stage for each output and a type
            std::unordered_map<std::shared_ptr<StageImpl>, std::unordered_map<std::string, std::stringstream>> MULIOperatorCallDefForStageAndType;
            std::unordered_map<std::string, std::string> INSERT_MULTI_OPERATOR_FOR_TYPE;
            //Computation function declaration for each output and type
            std::unordered_map<std::shared_ptr<ExpressionImpl>, std::unordered_map<std::string, std::stringstream>> operatorDeclForStageAndTypes;

            for (auto stage : computationStages) {
                std::unordered_map<std::string, std::stringstream> MULIOperatorCallDefForType, operatorDeclForTypes;
                MULIOperatorCallDefForStageAndType[stage] = std::unordered_map<std::string, std::stringstream>();
                operatorDeclForStageAndTypes[stage] = std::unordered_map<std::string, std::stringstream>();

                for (auto type : ncclSupportedTypes) {
                    MULIOperatorCallDefForStageAndType[stage][type] = std::stringstream();
                    operatorDeclForStageAndTypes[stage][type] = std::stringstream();
                }
            }

            for (auto type : ncclSupportedTypes) {
                INSERT_MULTI_OPERATOR_FOR_TYPE[type] = INSERT_MULTI_OPERATOR;
                INSERT_MULTI_OPERATOR_FOR_TYPE[type].replace(INSERT_MULTI_OPERATOR.find("<T>")+1,
                                                            1UL, type);
            }

            /*Add computation function to reduce_kernel.h*/
            std::map<std::shared_ptr<StageImpl>, std::tuple<CStruct, CStruct>> allReduceKernelStructs;
            allReduceKernelStructs = generateFusedBinOpCommCollCodeForNCCL(pipeline, pipelineStage, stageToMixedPrecisions);

            const std::string INSERT_REDUCE_KERNEL = "/*{INSERT REDUCE KERNEL HERE}*/";
            std::string reduceKernelH = readFile(NCCL_SRC_PATH+"collectives/device/reduce_kernel.h.in");
            std::string allCompsBody;
            for (auto out : computationStages) {
                allCompsBody += std::get<0>(allReduceKernelStructs[out]).body +"\n";
            }
            reduceKernelH = reduceKernelH.replace(reduceKernelH.find(INSERT_REDUCE_KERNEL),
                                                  INSERT_REDUCE_KERNEL.size(), allCompsBody);

            std::string allCompsHalfBody;
            for (auto out : computationStages) {
                allCompsHalfBody += std::get<1>(allReduceKernelStructs[out]).body +"\n";
            }
            //Create a specialized function for half and half2.
            reduceKernelH = reduceKernelH.replace(reduceKernelH.find(INSERT_REDUCE_KERNEL_HALF), 
                                                  INSERT_REDUCE_KERNEL_HALF.size(), allCompsHalfBody);
            
            writeFile(NCCL_DST_PATH+"collectives/device/reduce_kernel.h", reduceKernelH);

            std::map<std::shared_ptr<StageImpl>, std::stringstream> operatorDeclTemplateForStage;           
            for (auto stage : computationStages) {
                 operatorDeclTemplateForStage[stage] = std::stringstream();
                 if (onlyCommCollStageIsMixedPrec) {
                    operatorDeclTemplateForStage[stage]  << "__device__ " << vecTypeForStage[stage] << " " << std::get<0>(allReduceKernelStructs[stage]).name << "(";                      
                 } else {
                    operatorDeclTemplateForStage[stage]  << "__device__ PackType " << std::get<0>(allReduceKernelStructs[stage]).name << "(";
                 }

                 for (auto type : ncclSupportedTypes) {
                    operatorDeclForStageAndTypes[stage][type] << operatorDeclTemplateForStage[stage].str();
                 }
            }

            for (auto stage : computationStages) {
                InputsVisitor inVisitor;
                auto inputs = stage->usedExprs();

                for (auto iter = inputs.begin(); iter != inputs.end();) {
                    auto arg = *iter;
                    if (arg->type() == VariableNode || arg == reduceTensorStage) {
                        operatorDeclTemplateForStage[stage] << "const " << " T " << arg->name();
                        for (auto type : ncclSupportedTypes)
                            operatorDeclForStageAndTypes[stage][type] << "const " << type << " " << arg->name();
                    } else {
                        if (onlyCommCollStageIsMixedPrec) {
                            const size_t packTypeSize = elemsIn64Bits*sizeOfElemType(arg->elemType());
                            const std::string packType = uintTypeForSize(packTypeSize);
                            operatorDeclTemplateForStage[stage] << "const " << packType << " " << arg->name();
                            for (auto type : ncclSupportedTypes)
                                operatorDeclForStageAndTypes[stage][type] << "const " << packType << " " << arg->name();
                        } else {
                            operatorDeclTemplateForStage[stage] << "const PackType" << " " << arg->name();
                            for (auto type : ncclSupportedTypes)
                                operatorDeclForStageAndTypes[stage][type] << "const PackType" << " " << arg->name();
                        }
                    }

                    
                    if (++iter != inputs.end()) {
                        operatorDeclTemplateForStage[stage] << ", ";
                        for (auto type : ncclSupportedTypes)
                            operatorDeclForStageAndTypes[stage][type] << ", ";
                    }
                }

                // operatorDeclTemplateForStage[stage] << "const PackType" << " " << stage->name();
                // for (auto type : ncclSupportedTypes)
                //     operatorDeclForStageAndTypes[stage][type] << "const PackType" << " " << stage->name();
            }

            for (auto stage : computationStages) {
                operatorDeclTemplateForStage[stage] << ") const";
                for (auto type : ncclSupportedTypes)
                    operatorDeclForStageAndTypes[stage][type] << ") const";
            }

            std::string commonKernelH = readFile(NCCL_SRC_PATH+"collectives/device/common_kernel.h.in");

            std::string allOperatorDecls;
            for (auto stage : computationStages) {
                allOperatorDecls = allOperatorDecls + indent(1) + operatorDeclTemplateForStage[stage].str() + ";\n";
            }

            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_MULTI_OPERATOR),
                                                  INSERT_MULTI_OPERATOR.size(),
                                                  allOperatorDecls);
            std::stringstream packAllReduceOutputType;

            if (onlyCommCollStageIsMixedPrec) {
                /*For mixed precision, generate code in MULTI only for the type of AllReduce's output*/
                const std::string commCollStageCType = elemTypeToCType((*commCollStages.begin())->elemType());
                std::map<TensorElemType, std::string> elemTypeToConverter;
                std::map<std::string, std::vector<std::string>> converterToMembers;
                std::string packStructName = "FOO";

                for (auto stage : computationStages) {
                    for (auto cast : stageToMixedPrecisions[stage]) {
                        for (auto t : std::vector<TensorElemType>{cast->op()->elemType(), cast->elemType()}) {
                            if (elemTypeToConverter.count(t) > 0) 
                                continue;

                            const std::string ctype = elemTypeToCType(t);
                            std::vector<std::string> members;

                            if (t == Float16) {
                                packAllReduceOutputType << "struct converter" << ctype << "{";
                                for (size_t i = 0; i < elemsIn64Bits/2; i++) {
                                    packAllReduceOutputType << "half2 x"<< i << ";" << std::endl;
                                    members.push_back("x" + std::to_string(i));
                                    packAllReduceOutputType << indent(1) << "__device__ half getx" << 2*i << "(){ return __low2half(" << "x" << std::to_string(i) << ");}" << std::endl;
                                    packAllReduceOutputType << indent(1) << "__device__ half getx" << 2*i+1 << "(){ return __high2half(" << "x" << std::to_string(i) << ");}"  << std::endl;
                                }
                                packAllReduceOutputType << "};" << std::endl;
                            } else {
                                packAllReduceOutputType << indent(2) << "union converter" << ctype << "{" << std::endl;
                                packAllReduceOutputType << indent(3) << "uint64_t storage;" << std::endl;
                                packAllReduceOutputType << indent(3) << "struct {";
                                for (size_t i = 0; i < elemsIn64Bits; i++) {
                                    packAllReduceOutputType << ctype << " " << "x"<< i<< ";" << std::endl;
                                    members.push_back("x" + std::to_string(i));
                                    packAllReduceOutputType << "__device__ "<<ctype<<" getx" << i << "(){ return x" << std::to_string(i) << ";}" << std::endl;
                                }
                                packAllReduceOutputType << ";}" << packStructName<< ";" << std::endl <<"};" << std::endl;
                            }

                            elemTypeToConverter[t] = "converter" + ctype;
                            converterToMembers[elemTypeToConverter[t]] = members;
                        }
                    }
                }
                
                for (auto stage : computationStages) {
                    InputsVisitor inVisitor;
                    auto inputs = inVisitor.inputs(*stage);
                    TensorElemType t = stage->elemType();

                    if (sizeOfElemType(t) == 8) {
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] <<  indent(2) << elemTypeToCType(t) << " rv = FUNC()(";
                    }

                    for (auto arg : inputs) {
                        TensorElemType t = arg->elemType();

                        if (arg->type() == VariableNode) {
                            
                        } else {
                            if (t == Float16) {
                                MULIOperatorCallDefForStageAndType[stage]["half"] << elemTypeToConverter[t] << " c" << arg->name() << ";" << std::endl;
                                MULIOperatorCallDefForStageAndType[stage]["half"] << "c" << arg->name() << " = *(reinterpret_cast<const " << elemTypeToConverter[t] << "*>(&" << arg->name() << "));" << std::endl;
                            } else if (sizeOfElemType(t) == 4) {
                                MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << elemTypeToConverter[t] << " c" << arg->name() << ";" << std::endl; 
                                MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << "c" << arg->name() << ".storage = " << arg->name() << ";" << std::endl;
                            } else if (sizeOfElemType(t) == 8) {
                                if (t == Float64)
                                    MULIOperatorCallDefForStageAndType[stage]["double"] <<  "__longlong_as_double(" << arg->name() << ")";
                                else 
                                    MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << "(" << commCollStageCType << ")" << arg->name();
                            } else {
                                ASSERT(false, "Unknown type.");
                            }
                        }
                    }

                    if (sizeOfElemType(t) < 4) {
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << elemTypeToConverter[t] << " c" << stage->name() << ";" << std::endl;
                        std::map<ExpressionImpl*, std::vector<std::string>> membersMap;
                        std::stringstream body;

                        for (auto m : converterToMembers[elemTypeToConverter[t]]) {
                            body << indent(2) << "c" << stage->name() << "." << m << " = __halves2half2(FUNC()(";
                            for (auto i = inputs.begin(); i != inputs.end();) {
                                auto e = *i;
                                if (e->type() == VariableNode) {
                                    body << e->name();
                                } else {
                                    if (e->elemType() == Float16) {
                                        body << "__low2half(c" << e->name() << "." << m << ")";
                                    } else {
                                        body<< "c" << e->name() << "." << m;
                                    }
                                }

                                if (++i != inputs.end()) 
                                    body << ", ";
                            }
                            body << "), ";
                            body << "FUNC()(";
                            for (auto i = inputs.begin(); i != inputs.end();) {
                                auto e = *i;
                                if (e->type() == VariableNode) {
                                    body << e->name();
                                } else {
                                    if (e->elemType() == Float16) {
                                        body << "__high2half(c" << e->name() << "." << m << ")";
                                    } else {
                                        body<< "c" << e->name() << "." << m;
                                    }
                                }

                                if (++i != inputs.end()) 
                                    body << ", ";
                            }
                            body << "));" << std::endl;
                        }

                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << body.str();
                    }

                    if (sizeOfElemType(t) == 4) {
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << elemTypeToConverter[t] << " c" << stage->name() << ";" << std::endl;
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << MULTIMethodBodyFor4BytesTypeMixedPrecision(stage, inputs, pipelineStage, commCollStageCType, 
                                                                                                                                    converterToMembers[elemTypeToConverter[t]], false,
                                                                                                                                    packStructName);
                    }
                }

                for (auto stage : computationStages) {
                    if (stage->elemType() == Float16) {
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << "return *(reinterpret_cast<" << uintTypeForSize(elemsIn64Bits*sizeOfElemType(stage->elemType())) <<"*>(&c" << stage->name() << "));" << std::endl;
                    } else if (sizeOfElemType(stage->elemType()) == 4)
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << "return c" << stage->name() << ".storage;" << std::endl;
                    else if (sizeOfElemType(stage->elemType()) == 8) {
                        MULIOperatorCallDefForStageAndType[stage][commCollStageCType] << ");" << std::endl
                                                                                      << indent(2) << "return rv;"<< std::endl;
                    }
                }
            } else {
                /*When mixed precision is not used then generate code for all types*/
                //Generate code for float, int32_t, and unit32_t
                for (auto stage : computationStages) {
                    auto inputs = stage->usedExprs();
                    
                    for (auto arg : inputs) {
                        if (arg->type() == VariableNode || arg == reduceTensorStage) {
                            
                        } else {
                            for (auto t : std::vector<std::string>{"float", "int32_t", "uint32_t"}) {
                                MULIOperatorCallDefForStageAndType[stage][t] << "converter c" << arg->name() << ";" << std::endl;
                                MULIOperatorCallDefForStageAndType[stage][t] << "c" << arg->name() << ".storage = " << arg->name() << ";" << std::endl;
                            }

                            MULIOperatorCallDefForStageAndType[stage]["half"] << "struct PackHalf2 c" << arg->name() << ";" << std::endl;
                            MULIOperatorCallDefForStageAndType[stage]["half"] << "c" << arg->name() << " = *(reinterpret_cast<const struct PackHalf2*>(&" << arg->name() << "));" << std::endl;
                        }
                    }

                    for (auto t : std::vector<std::string>{"float", "int32_t", "uint32_t"}) {
                        MULIOperatorCallDefForStageAndType[stage][t] << "converter c" << stage->name() << ";" << std::endl;
                        MULIOperatorCallDefForStageAndType[stage][t] << MULTIMethodBodyFor4BytesType(stage, inputs, pipelineStage, t, std::vector<std::string>{"a", "b"}, true);
                    }
                    MULIOperatorCallDefForStageAndType[stage]["half"] << "struct PackHalf2 c" << stage->name() << ";" << std::endl;
                    MULIOperatorCallDefForStageAndType[stage]["half"] << MULTIMethodBodyFor4BytesType(stage, inputs, pipelineStage, "half", std::vector<std::string>{"a", "b"}, true);
                }

                for (auto stage : computationStages) {
                    for (auto t : std::vector<std::string>{"float", "int32_t", "uint32_t"}) {
                        MULIOperatorCallDefForStageAndType[stage][t] << "return c" << stage->name() << ".storage;" << std::endl;
                    }
                    MULIOperatorCallDefForStageAndType[stage]["half"] << "return *(reinterpret_cast<PackType*>(&c" << stage->name() << "));" << std::endl;
                }

                //Generate code for int64_t, uint64_t, and double
                for (auto stage : computationStages) {
                    MULIOperatorCallDefForStageAndType[stage]["double"] <<  indent(2) <<"double rv = FUNC()(";
                    MULIOperatorCallDefForStageAndType[stage]["int64_t"] <<  indent(2) <<"int64_t rv = FUNC()(";
                    MULIOperatorCallDefForStageAndType[stage]["uint64_t"] <<  indent(2) <<"uint64_t rv = FUNC()(";
                    InputsVisitor inVisitor;
                    auto inputs = inVisitor.inputs(*stage);

                    for (auto iter = inputs.begin(); iter != inputs.end();) {
                        auto arg = *iter;
                        if (arg->type() == VariableNode) {
                            MULIOperatorCallDefForStageAndType[stage]["double"] << arg->name();
                            MULIOperatorCallDefForStageAndType[stage]["int64_t"] << arg->name();
                        } else {
                            MULIOperatorCallDefForStageAndType[stage]["double"] <<  "__longlong_as_double(" << arg->name() << ")";
                            MULIOperatorCallDefForStageAndType[stage]["int64_t"] << "(int64_t)" << arg->name();
                        }

                        MULIOperatorCallDefForStageAndType[stage]["uint64_t"] << arg->name();

                        if (++iter != inputs.end()) {
                            MULIOperatorCallDefForStageAndType[stage]["double"] << ", ";
                            MULIOperatorCallDefForStageAndType[stage]["uint64_t"] << ", ";
                            MULIOperatorCallDefForStageAndType[stage]["int64_t"] << ", ";
                        }
                    }

                    MULIOperatorCallDefForStageAndType[stage]["double"] << ");"<< std::endl;
                    MULIOperatorCallDefForStageAndType[stage]["int64_t"] << ");"<< std::endl;
                    MULIOperatorCallDefForStageAndType[stage]["uint64_t"] << ");"<< std::endl;

                    MULIOperatorCallDefForStageAndType[stage]["double"] << indent(2) << "return rv;"<< std::endl;
                    MULIOperatorCallDefForStageAndType[stage]["int64_t"] << indent(2) << "return rv;"<< std::endl;
                    MULIOperatorCallDefForStageAndType[stage]["uint64_t"] << indent(2) << "return rv;" << std::endl;
                }
            }

            for (auto type : ncclSupportedTypes) {
                std::string allFuncsForAType;

                allFuncsForAType += packAllReduceOutputType.str() +"\n";
                for (auto stage : computationStages) {
                    allFuncsForAType += indent(1) + operatorDeclForStageAndTypes[stage][type].str() + " {\n" + MULIOperatorCallDefForStageAndType[stage][type].str() + "}\n";
                }

                commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_MULTI_OPERATOR_FOR_TYPE[type]),
                                                      INSERT_MULTI_OPERATOR_FOR_TYPE[type].size(),
                                                      allFuncsForAType);
            }

            writeFile(NCCL_DST_PATH+"collectives/device/common_kernel.h", commonKernelH);
            
            /*Write to prims_ll_computation.h*/
            const std::string INSERT_ReduceCopySend_ARGS = "/*{INSERT ReduceCopySend ARGS}*/";
            const std::string INSERT_LLGenericOp2_ARGS = "/*{INSERT LLGenericOp2 ARGS}*/";
            const std::string INSERT_LLGenericOp2_CALL_PARAMS = "/*{INSERT LLGenericOp2 CALL PARAMS}*/";
            const std::string INSERT_ALLREDUCE_ARG_PACK = "/*{INSERT ALLREDUCE ARG PACK}*/";
            const std::string INSERT_OUTPUT_PACK = "/*{INSERT OUTPUT PACK}*/";
            const std::string INSERT_ARGS_WITH_CAST = "/*{INSERT ARGS WITH CAST}*/";
            const std::string INSERT_COMPUTATION = "/*{INSERT COMPUTATION}*/";
            const std::string INSERT_OUTPUT_VAL = "/*{INSERT OUTPUT VAL}*/";
            const std::string INSERT_Recv_ARGS = "/*{INSERT recv ARGS}*/";
            const std::string INSERT_LLGenericOp2_CALL_PARAMS_FOR_recv = "/*{INSERT LLGenericOp2 CALL PARAMS FOR recv}*/";
            const std::string INSERT_ALLGATHER_COMPUTATION = "/*{INSERT ALLGATHER COMPUTATION}*/";
            const std::string INSERT_ALLGATHER_OUTPUT_PACK = "/*{INSERT ALLGATHER OUTPUT PACK}*/";
            const std::string INSERT_ALLGATHER_OUTPUT_VAL = "/*{INSERT ALLGATHER OUTPUT VAL}*/";
            const std::string INSERT_NOT_ALLGATHER_COMPUTE = "/*{INSERT NOT ALLGATHER_COMPUTE}*/";
            const std::string INSERT_FINAL_OUTPUT_PACK = "/*{INSERT FINAL OUTPUT PACK}*/";
            const std::string INSERT_FINAL_OUTPUT_VAL = "/*{INSERT FINAL OUTPUT VAL}*/";
            const std::string INSERT_FINAL_COMPUTATION = "/*{INSERT FINAL COMPUTATION}*/";

            //Update signatures of LLprims::recvReduceCopySend
            std::stringstream recvReduceCopySendDecl;
            std::stringstream LLGenericOp2Args;
            std::stringstream LLGenericOp2CallParams;
            std::stringstream primsLL128RecvReduceSendCopyArgs;
            std::stringstream primsLL128RecvReduceSendCopyCallParams;
            std::stringstream primsLLRecvArgs;
            std::stringstream LLGenericOp2CallParamsForRecv2;
            std::stringstream primsLLAllGatherComputation;
            std::stringstream allGatherOutputPack;

            if (fusedToAllReduce) {
                primsLLRecvArgs << printArgumentForNCCLPrims((*outputs.begin())) << ", ";
                
                for (auto it = gpuKernelArgs.begin(); it != gpuKernelArgs.end(); ++it) {
                    if (outputs.count(*it) > 0) {
                        LLGenericOp2CallParamsForRecv2 << (*outputs.begin())->name() << ", ";
                    } else {
                        if ((*it)->type() == VariableNode) {
                            LLGenericOp2CallParamsForRecv2 << "(T)0, ";    
                        } else {
                            LLGenericOp2CallParamsForRecv2 << "nullptr, ";
                        }
                    }
                }

                allGatherOutputPack << (*outputs.begin())->name() << "Pack";
            }

            for (auto it = gpuKernelArgs.begin(); it != gpuKernelArgs.end(); ++it) {
                bool isInputToCommColl = (*it) == commCollArg;
                recvReduceCopySendDecl << ((isInputToCommColl) ? "const " : "") + printArgumentForNCCLPrims(*it);
                LLGenericOp2Args << ((isInputToCommColl) ? "const " : "") + printArgumentForNCCLPrims(*it);
                LLGenericOp2CallParams << (*it)->name();
                recvReduceCopySendDecl << ", ";
                LLGenericOp2Args << ", ";
                LLGenericOp2CallParams << ", ";
                if ((*it)->type() != VariableNode) {
                    primsLL128RecvReduceSendCopyArgs << "uint64_t* " << (*it)->name() << "Pack";
                    primsLL128RecvReduceSendCopyCallParams << (*it)->name() << "Pack";
                }
                else {
                    primsLL128RecvReduceSendCopyArgs << printArgumentForNCCLPrims(*it);
                    primsLL128RecvReduceSendCopyCallParams << (*it)->name();
                }
                primsLL128RecvReduceSendCopyArgs << ", ";
                primsLL128RecvReduceSendCopyCallParams << ", ";
            }

            std::shared_ptr<ExpressionImpl> allReduceArg = commCollArg;
            
            std::stringstream LLGenericOp2ArgsCast;

            for (auto arg : gpuKernelArgs) {
                //Add "Pack" for only non Variable Nodes
                if (arg->type() != VariableNode) {
                    LLGenericOp2ArgsCast << "uint64_t* " << arg->name() << "Pack = " << "(uint64_t*)" << arg->name() << ";" << std::endl;
                }
            }

            std::stringstream primsComputation;
            std::stringstream primsLL128AllGatherComputationVU;
            std::stringstream primsLL128AllGatherComputationVUPlus1;
            std::stringstream primsLL128ComputationVU;
            std::stringstream primsLL128ComputationVUPlus1;

            if (!hasReduceTensor) {
                for (auto stage : computationStages) {                    
                    std::stringstream stageComputation;
                    std::stringstream primsLL128StageComputationVU, primsLL128StageComputationVUPlus1;

                    stageComputation << indent(2) << "uint64_t " << stage->name() << " = MULTI<" << std::get<0>(allReduceKernelStructs[stage]).name<< "<T>" << ", T>()." <<  std::get<0>(allReduceKernelStructs[stage]).name << "(";
                    std::string outputNameVU, outputNameVUPlus1;
                    if (fusedToAllReduce and stage == AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()) {
                        outputNameVU = "v[u]";
                        outputNameVUPlus1 = "v[u+1]";
                    } else {
                        outputNameVU = (outputStages.count(stage) == 0 ? "uint64_t " + stage->name() : "v[u]");
                        outputNameVUPlus1 = (outputStages.count(stage) == 0 ? "uint64_t " + stage->name() : "v[u + 1]");
                    }
                    primsLL128StageComputationVU << indent(2) << outputNameVU << " = MULTI<" << std::get<0>(allReduceKernelStructs[stage]).name<< "<T>" << ", T>()." <<  std::get<0>(allReduceKernelStructs[stage]).name << "(";
                    primsLL128StageComputationVUPlus1 << indent(2) << outputNameVUPlus1 << " = MULTI<" << std::get<0>(allReduceKernelStructs[stage]).name<< "<T>" << ", T>()." <<  std::get<0>(allReduceKernelStructs[stage]).name << "(";

                    //FIXME: To the call to MULTI, we add all arguments other than 
                    //the result of Stage and the argument of AllReduce.
                    auto inputs = stage->usedExprs();
                    bool hasAllGatherStageAsArg = false;

                    for (auto iter = inputs.begin(); iter != inputs.end();) {
                        auto arg = *iter;
                        if (commCollOutput == arg) {
                            stageComputation << "val"; 
                            primsLL128StageComputationVU << "v[u]";
                            primsLL128StageComputationVUPlus1 << "v[u+1]";
                        } else {
                            //If it is an argument to pipe then we emit "+ offset" too.
                            //We do not check for explicit Store locations right now to transfer data through register.
                            if (arg->type() == TensorNode) {
                                if (outputs.count(arg) > 0) {
                                    primsLL128StageComputationVU << "shmem64Ptr[u*(WARP_SIZE-2)]";
                                    primsLL128StageComputationVUPlus1 << "shmem64Ptr[u*(WARP_SIZE-2)+1]";
                                } else {
                                    primsLL128StageComputationVU << "*(" << arg->name() << "Pack + u*(WARP_SIZE))";    
                                    primsLL128StageComputationVUPlus1 << "*(" << arg->name() << "Pack + u*(WARP_SIZE)+1)";    
                                }
                                stageComputation << "*(" << arg->name() << "Pack + offset)";
                            } else {
                                if (fusedToAllReduce and arg == allGatherStage) {
                                    //AllGather stage is now the received value in LLprims.recv
                                    hasAllGatherStageAsArg = true;
                                    stageComputation << "%s";
                                    primsLL128StageComputationVU << "v[u]";
                                    primsLL128StageComputationVUPlus1 << "v[u+1]";
                                } else {
                                    stageComputation << arg->name();
                                    primsLL128StageComputationVU << arg->name();
                                    primsLL128StageComputationVUPlus1 << arg->name();
                                }
                            }
                        }

                        if (++iter != inputs.end()) {
                            stageComputation << ", ";
                            primsLL128StageComputationVU << ", ";
                            primsLL128StageComputationVUPlus1 << ", ";
                        }
                    }

                    stageComputation << ");" << std::endl;
                    primsLL128StageComputationVU << ");" << std::endl;
                    primsLL128StageComputationVUPlus1 << ");" << std::endl;
                    
                    char storeCodeFmt[] = "if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {\n"
                                        "    // Last incomplete word\n"
                                        "    storeAL(%sPack+offset, %s, nbytes & 0x7);\n"
                                        "} else {\n"
                                        "    storeAL(%sPack+offset, %s, sizeof(uint64_t));\n"
                                        "}\n";

                    //In case of fusing ReduceScatter+AllGather into AllReduce, store the input of 
                    //AllGather to the input of ReduceScatter.
                    // if (fusedToAllReduce and 
                    //     stage == AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg().get()) {
                    //     ExpressionImpl* allreduceInput = AstNodeImpl::asReduceScatterImpl(reduceScatterStage->definition())->arg().get();
                    //     char storeCode[1024];
                    //     sprintf(storeCode, storeCodeFmt, allreduceInput->name().c_str(), stage->name().c_str(), 
                    //             allreduceInput->name().c_str(), stage->name().c_str());

                    //     primsComputation << storeCode << std::endl;
                    // }
                    ASSERT(false, "FIX");
                    //FIX
                    // if (pipeline.explicitStoreLocations().count(stage) > 0 and 
                    //     (pipeline.explicitStoreLocations().count(stage) > 0 ? outputs.count(pipeline.explicitStoreLocations().at(stage)) == 0 : 
                    //                                                         outputs.count(stage) == 0)) {
                    //     //If explicitly stored at a location then store it there 
                    //     const char* nameStr = pipeline.explicitStoreLocations().at(stage)->name().c_str();
                    //     char storeCode[1024];
                    //     sprintf(storeCode, storeCodeFmt, nameStr, stage->name().c_str(), nameStr, stage->name().c_str());

                    //     stageComputation << storeCode << std::endl;
                    // }

                    if (fusedToAllReduce) {
                        //If ReduceScatter and AllGather are fused to AllReduce then do not generate computation for the output.
                        bool isStageAfterAllGather = std::find(stagesAfterAllGather.begin(), stagesAfterAllGather.end(), stage) != stagesAfterAllGather.end();
                        if (outputStages.count(stage) > 0 || isStageAfterAllGather) {
                            if (hasAllGatherStageAsArg) {
                                std::string ss = stageComputation.str();
                                 primsLLAllGatherComputation << replaceAllSubString(ss, "%s", "val");
                            } else {
                                primsLLAllGatherComputation << stageComputation.str();
                            }
                            if (outputStages.count(stage) > 0) {
                                std::string ss = primsLL128StageComputationVU.str();
                                replaceAllSubString(ss, "v[u] = ", "v1[u] = ");
                                primsLL128AllGatherComputationVU << ss;
                                ss = primsLL128StageComputationVUPlus1.str();
                                replaceAllSubString(ss, "v[u + 1] = ", "v1[u + 1] = ");
                                primsLL128AllGatherComputationVUPlus1 << ss;
                            } else {
                                primsLL128AllGatherComputationVU << primsLL128StageComputationVU.str();
                                primsLL128AllGatherComputationVUPlus1 << primsLL128StageComputationVUPlus1.str();
                            }
                        }
                        if (hasAllGatherStageAsArg) {
                            std::string ss = stageComputation.str();
                            primsComputation << replaceAllSubString(ss, "%s", AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()->name());
                        } else {
                            primsComputation << stageComputation.str();
                        }

                        primsLL128ComputationVU << primsLL128StageComputationVU.str();
                        primsLL128ComputationVUPlus1 << primsLL128StageComputationVUPlus1.str();
                        

                    } else {
                        primsComputation << stageComputation.str();
                        primsLL128ComputationVU << primsLL128StageComputationVU.str();
                        primsLL128ComputationVUPlus1 << primsLL128StageComputationVUPlus1.str();
                    }
                }
            } else {
                ASSERT(false, "FIX");
                //FIX
                // for (auto stage : computationStages) {
                //     if (pipeline.explicitStoreLocations().count(stage) > 0 and  outputs.count(pipeline.explicitStoreLocations().at(stage))  == 1)
                //         primsComputation << indent(2) << "uint64_t " << stage->name() << " = 0;" << std::endl;
                // }
            }
            
            std::string primsLLH = readFile(NCCL_SRC_PATH+"collectives/device/prims_ll_computation.h.in");
            if (fusedToAllReduce) {
                primsLLH = replaceAllSubString(primsLLH, INSERT_LLGenericOp2_CALL_PARAMS_FOR_recv, 
                                            LLGenericOp2CallParamsForRecv2.str());
                primsLLH = primsLLH.replace(primsLLH.find(INSERT_Recv_ARGS), INSERT_Recv_ARGS.size(),
                                            primsLLRecvArgs.str());
                primsLLH = primsLLH.replace(primsLLH.find(INSERT_Recv_ARGS), INSERT_Recv_ARGS.size(),
                                            primsLLRecvArgs.str());
            }
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_ReduceCopySend_ARGS),
                                        INSERT_ReduceCopySend_ARGS.size(), recvReduceCopySendDecl.str());
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_LLGenericOp2_ARGS),
                                        INSERT_LLGenericOp2_ARGS.size(), LLGenericOp2Args.str());
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_LLGenericOp2_CALL_PARAMS),
                                        INSERT_LLGenericOp2_CALL_PARAMS.size(), LLGenericOp2CallParams.str());
            primsLLH = replaceAllSubString(primsLLH, INSERT_ALLREDUCE_ARG_PACK,
                                           allReduceArg->name() + "Pack");
            if (fusedToAllReduce) {
                primsLLH = replaceAllSubString(primsLLH, INSERT_OUTPUT_PACK,
                                               commCollArg->name() + "Pack");
            } else {
                primsLLH = replaceAllSubString(primsLLH, INSERT_OUTPUT_PACK,
                                               (*outputs.begin())->name() + "Pack");
            }
            primsLLH = replaceAllSubString(primsLLH, INSERT_FINAL_OUTPUT_PACK,
                                           (*outputs.begin())->name() + "Pack");
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_ARGS_WITH_CAST),
                                        INSERT_ARGS_WITH_CAST.size(), LLGenericOp2ArgsCast.str());
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_COMPUTATION),
                                        INSERT_COMPUTATION.size(), primsComputation.str());
            if (fusedToAllReduce) {
                primsLLH = replaceAllSubString(primsLLH, INSERT_OUTPUT_VAL, AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()->name());
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_COMPUTATION, primsLLAllGatherComputation.str());
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_OUTPUT_PACK, (*outputs.begin())->name() + "Pack");
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_OUTPUT_VAL, (*outputStages.begin())->name());
                //Prevent storing and only send in recvReduceCopySend.
                primsLLH = replaceAllSubString(primsLLH, INSERT_NOT_ALLGATHER_COMPUTE, "&& false");
            } else {
                //Generate same code for ALLGATHER OUTPUT 
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_COMPUTATION, primsLLAllGatherComputation.str());
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_OUTPUT_PACK, (*outputs.begin())->name() + "Pack");
                primsLLH = replaceAllSubString(primsLLH, INSERT_ALLGATHER_OUTPUT_VAL, "0");
                primsLLH = replaceAllSubString(primsLLH, INSERT_OUTPUT_VAL, (*outputStages.begin())->name());
            }
            
            primsLLH = replaceAllSubString(primsLLH, INSERT_FINAL_OUTPUT_VAL, (*outputStages.begin())->name());
            
            writeFile(NCCL_DST_PATH+"collectives/device/prims_ll_computation.h", primsLLH);
            
            /*Write to prims_ll128_computation.h*/
            const std::string PRIMSLL128_RECV_REDUCE_COPY_ARGS = "/*{INSERT recvReduceCopy ARGS}*/";
            const std::string PRIMSLL128_GenericOp2_CALL_PARAMS = "/*{INSERT GenericOp2 CALL PARAMS}*/";
            const std::string PRIMSLL128_GenericOp2_CALL_PARAMS_FOR_ALLGATHER_COMPUTE = "/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/";
            const std::string PRIMSLL128_RecvReduceCopySend_ARGS = "/*{INSERT RecvReduceCopySend ARGS}*/";
            const std::string PRIMSLL128_GenericOp2_CALL_ARGS = "/*PRIMSLL128 {INSERT GenericOp2 ARGS}*/";
            const std::string PRIMSLL128_GenericOp2_ARGS_CAST = "/*{INSERT ARGS WITH CAST}*/";
            const std::string PRIMSLL128_ALLREDUCE_ARG = "/*{INSERT ALLREDUCE ARG}*/";
            const std::string PRIMSLL128_ALLREDUCE_ARG_PACK = "/*{INSERT ALLREDUCE ARG PACK}*/";
            const std::string PRIMSLL128_OUTPUT = "/*{INSERT OUTPUT}*/";
            const std::string PRIMSLL128_OUTPUT_PACK = "/*{INSERT OUTPUT PACK}*/";
            const std::string PRIMSLL128_COMPUTATION = "/*PRIMSLL128: {INSERT COMPUTATION}*/";
            const std::string PRIMSLL128_ALLGATHER_COMPUTATION = "/*PRIMSLL128: {INSERT ALLGATHER COMPUTATION}*/";
            const std::string PRIMSLL128_RecvReduceSendCopy_ARGS = "/*PRIMSLL128 {INSERT recvReduceSendCopy2 ARGS}*/";
            const std::string PRIMSLL128_RecvReduceCopy2_CALL_PARAMS = "/*{INSERT recvReduceSendCopy2 CALL PARAMS}*/";
            const std::string PRIMSLL128_RECV_ARGS = "/*PRIMSLL128: {INSERT recv ARGS}*/";

            std::string primsLLH128 = readFile(NCCL_SRC_PATH+"collectives/device/prims_ll128_computation.h.in");

            std::string outputName;

            if (false && fusedToAllReduce) {
                //If fused then output to ReduceScatter's argument;
                outputName = AstNodeImpl::asReduceScatterImpl(reduceScatterStage->definition())->arg()->name();
            } else {
                outputName = (*outputs.begin())->name();
            }

            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_RECV_REDUCE_COPY_ARGS, recvReduceCopySendDecl.str());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_RecvReduceCopySend_ARGS, 
                                              recvReduceCopySendDecl.str());
            if (fusedToAllReduce) {
                primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_GenericOp2_CALL_PARAMS_FOR_ALLGATHER_COMPUTE,
                                                LLGenericOp2CallParamsForRecv2.str());
                primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_RECV_ARGS, primsLLRecvArgs.str());
            }
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_GenericOp2_CALL_PARAMS,
                                              LLGenericOp2CallParams.str());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_GenericOp2_CALL_ARGS, 
                                              LLGenericOp2Args.str());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_GenericOp2_ARGS_CAST,
                                              LLGenericOp2ArgsCast.str());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_ALLREDUCE_ARG,
                                              commCollArg->name());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_ALLREDUCE_ARG_PACK,
                                              commCollArg->name() +"Pack");
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_OUTPUT,
                                              outputName);
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_OUTPUT_PACK,
                                              outputName + "Pack");
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_COMPUTATION,
                                              primsLL128ComputationVU.str() + "\n if (!flagThread) {\n" +
                                              primsLL128ComputationVUPlus1.str() + "}");
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_RecvReduceSendCopy_ARGS,
                                              primsLL128RecvReduceSendCopyArgs.str());
            primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_RecvReduceCopy2_CALL_PARAMS,
                                              primsLL128RecvReduceSendCopyCallParams.str());
            if (fusedToAllReduce) {
                primsLLH128 = replaceAllSubString(primsLLH128, PRIMSLL128_ALLGATHER_COMPUTATION, 
                                                  primsLL128AllGatherComputationVU.str() + "\n if (!flagThread) {\n" +
                                                  primsLL128AllGatherComputationVUPlus1.str() + "}");
            }
            writeFile(NCCL_DST_PATH+"collectives/device/prims_ll128_computation.h", primsLLH128);

            char amongGPUReductionCode[1024];
            std::stringstream afterReductionComputation;
            std::stringstream packVariablesDecl;
            std::stringstream reduceTensorDAGComputation;
            std::string reduceTensorAtomicOpName = "atomic";
            ReduceTensorImpl* reduceTensorNode = nullptr;
            const size_t packFactor = (hasReduceTensor) ? sizeof(uint64_t)/sizeOfElemType(reduceTensorStage->elemType()) : 0;
            std::string atomicOpInput;
            if (hasReduceTensor) {
                reduceTensorNode = AstNodeImpl::asReduceTensorImpl(reduceTensorStage->definition()).get();

                switch (reduceTensorNode->op()) {
                    case Summation:
                        reduceTensorAtomicOpName += "Add";
                        break;
                    
                    default:
                        ASSERT(false, "Unimplemented reduction operation");
                }
            
                if (reduceTensorNode->arg() == commCollOutput) {
                    atomicOpInput = "val";
                } else {
                    atomicOpInput = reduceTensorNode->arg()->name();
                }
            }
            if (hasReduceTensor) {
                //If there is a reduce tensor operation, then first generate all computations required to 
                //get the result of reduction.
                //All computations (include the atomic Ops) are done using 64-bits.
                for (auto arg : gpuKernelArgs) {
                    if (arg->type() == VariableNode)
                        continue;

                    packVariablesDecl << indent(2) << "uint64_t* " << arg->name() << "Pack = " << "(uint64_t*)"
                                      << "(" << arg->name() << "+ offset)" << ";" << std::endl;
                }
                
                for (auto stage : reduceTensorDAG) {
                    if (stage == commCollOutput) {
                        reduceTensorDAGComputation << indent(2) << "uint64_t val = " << "*(" + commCollArg->name() + "Pack + offset);"
                                                   << std::endl;
                        continue;
                    }

                    if (stage->definition()->type() == ReduceTensorNode) {
                        continue;
                    }
                    
                    reduceTensorDAGComputation << generateStageCompUsingMULTI(pipeline, stage, commCollOutput, std::get<0>(allReduceKernelStructs[stage]).name,
                                                                              "val") 
                                               << std::endl;
                }

                std::string allReduceH = readFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h");

                std::stringstream computationLoop;
                computationLoop << indent(1) << "for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {"<<std::endl
                                    << indent(2) << "chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);" <<std::endl
                                    << indent(2) << "ssize_t offset; int nelem; int chunk;" <<std::endl
                                    << indent(2) << "chunk = ring->devUserRanks[0];" <<std::endl
                                    << indent(2) << "offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;"<<std::endl
                                    << indent(2) << "nelem = min(chunkSize, size-offset);"<<std::endl
                                    << packVariablesDecl.str()<<std::endl;
                
                perGPUReductionComp << computationLoop.str()
                                    << indent(2) << "for (ssize_t offset = threadIdx.x; offset < nelem" << "/" << packFactor << "; offset += blockDim.x){" << std::endl
                                    << reduceTensorDAGComputation.str() 
                                    << indent(3) << "MULTI<FUNC, T>()." << reduceTensorAtomicOpName << "(&" << reduceTensorStage->name().c_str() << ", " << atomicOpInput << ");"<< std::endl
                                    << indent(2) << "}" << std::endl
                                    << indent(1) << "}" << std::endl;
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_REDUCTION_END_LOOP), 
                                                RINGLL_REDUCTION_END_LOOP.size(), indent(1) +"}\n");
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_REDUCTION_PER_GPU), 
                                                RINGLL_REDUCTION_PER_GPU.size(), perGPUReductionComp.str());
                char amongGPURedFormat[] =  "if(threadIdx.x + blockDim.x*blockIdx.x == 0) args->comm->reduced%s = 0;\n"\
                                            "cooperative_groups::grid_group __grid_group = cooperative_groups::this_grid();\n"\
                                            "__grid_group.sync();\n"\
                                            "LLprims.send(&%s, 1);\n"\
                                            "for (int j=2; j<nranks; ++j) {\n"\
                                            "    LLprims.recvReduceSend(&%s, 1);\n"\
                                            "}\n"\
                                            "LLprims.recvReduceCopy(&%s, &%s, 1);\n"\
                                            "__grid_group.sync();\n"\
                                            "if(threadIdx.x == 0) {\n"\
                                            "    ::atomicAdd(&args->comm->reduced%s, (%s)%s);\n"\
                                            "}\n"\
                                            "__grid_group.sync();\n"\
                                            "%s = args->comm->reduced%s;\n";
                
                sprintf(amongGPUReductionCode, amongGPURedFormat, reduceTensorStage->name().c_str(), reduceTensorStage->name().c_str(), 
                        reduceTensorStage->name().c_str(), reduceTensorStage->name().c_str(), reduceTensorStage->name().c_str(),
                        reduceTensorStage->name().c_str(), elemTypeToCType(reduceTensorStage->elemType()).c_str(), reduceTensorStage->name().c_str(),
                        reduceTensorStage->name().c_str(), reduceTensorStage->name().c_str());
                std::stringstream reductionTransfer;
                int l = 1;
                reductionTransfer << amongGPUReductionCode << std::endl;
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_REDUCTION_TRANSFER),
                                                RINGLL_REDUCTION_TRANSFER.size(), reductionTransfer.str());

                afterReductionComputation << indent(2) << "uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);"  << std::endl
                                          << indent(2) << "uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));" << std::endl
                                          << indent(2) << "for (ssize_t offset = threadIdx.x; offset < nelem" << "/" << packFactor << "; offset += blockDim.x){" << std::endl;

                //After computation of reduction, perform all computations that are not in ReduceTensor Node's DAG

                for (auto stage : afterReduceTensorDAG) {
                    if (stage == commCollOutput) {
                        afterReductionComputation << indent(2) << "uint64_t val = " << "*(" + commCollArg->name() + "Pack + offset);"
                                                   << std::endl;
                        continue;
                    }

                    // TODO: being able to compute values only once would be ideal, but since the reduceTensorDAG and the after reduction part
                    // do are in separate loop bodies, we cannot just skip the 
                    // if (std::find(reduceTensorDAG.begin(), reduceTensorDAG.end(), stage) != reduceTensorDAG.end()) {
                    //     continue;
                    // }
                    
                    afterReductionComputation << generateStageCompUsingMULTI(pipeline, stage, commCollOutput, std::get<0>(allReduceKernelStructs[stage]).name,
                                                                             "val") 
                                              << std::endl;    
                    ASSERT(false, "FIX");
                    // FIX
                    // if (pipeline.explicitStoreLocations().count(stage) > 0 and 
                    //     (pipeline.explicitStoreLocations().count(stage) > 0 ? outputs.count(pipeline.explicitStoreLocations().at(stage)) == 1 : 
                    //                                                           outputs.count(stage) == 1)) {
                    //     //If explicitly stored at a location then store it there 
                    //     char storeCodeFmt[] = "if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {\n"
                    //                     "    // Last incomplete word\n"
                    //                     "    memcpy((char*)(%sPack+offset), (char*)&%s, nbytes & 0x7);\n"
                    //                     "} else {\n"
                    //                     "    memcpy((char*)(%sPack+offset), (char*)&%s, sizeof(uint64_t));\n"
                    //                     "}\n";
                    //     const char* nameStr = pipeline.explicitStoreLocations().at(stage)->name().c_str();
                    //     char storeCode[1024];
                    //     sprintf(storeCode, storeCodeFmt, nameStr, stage->name().c_str(), nameStr, stage->name().c_str());

                    //     afterReductionComputation << storeCode << std::endl;
                    // }                
                }

                afterReductionComputation << indent(1) << "}";

                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_REDUCTION_COMPUTATION),
                                                RINGLL_REDUCTION_COMPUTATION.size(), computationLoop.str() + afterReductionComputation.str() + 
                                                "LLprims.send(" + (*outputs.begin())->name() + " + offset, nelem);\n"); 
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_SHMEM_DECL_FOR_REDUCTION),
                                                RINGLL_SHMEM_DECL_FOR_REDUCTION.size(), "__shared__ T " + reduceTensorStage->name() + ";\n" + 
                                                "if (threadIdx.x == 0) " + reduceTensorStage->name() + " = 0;\n__syncthreads();\n");

                writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceH);

                const std::string NCCL_COMM_INSERT_REDUCE_VAL = "/*{INSERT REDUCE VAL}*/";

                std::string comm = readFile(NCCL_DST_PATH + "include/devcomm.h");
                comm = comm.replace(comm.find(NCCL_COMM_INSERT_REDUCE_VAL), NCCL_COMM_INSERT_REDUCE_VAL.size(), 
                                    elemTypeToCType(reduceTensorStage->elemType()) + " reduced" +  reduceTensorStage->name() + ";");
                writeFile(NCCL_DST_PATH + "include/devcomm.h", comm);
            }

            std::stringstream mixedPrecisionComputation;
            if (onlyCommCollStageIsMixedPrec) {
                //For Mixed precision add computation to RingLL
                const std::string RINGLL_INSERT_MIXED_PREC_COMPUT = "/*RINGLL: {INSERT MIXED-PRECISION COMPUTATION}*/";
                const std::string RINGLL128_INSERT_MIXED_PREC_COMPUT = "/*RINGLL128: {INSERT MIXED-PRECISION COMPUTATION}*/";
                mixedPrecisionComputation << std::endl << indent(2) << "if (nelem > 0) {" << std::endl;

                mixedPrecisionComputation << indent(3) << "const size_t packFactor = " << elemsIn64Bits << ";" << std::endl;
                for (auto arg : gpuKernelArgs) {
                    if (arg->type() == VariableNode) {
                        
                    } else {
                        std::string packType = (arg->elemType() == commCollOutput->elemType()) ? vecTypeName : "uint64_t";
                        mixedPrecisionComputation << indent(3) << packType << "* " << arg->name() << "Pack = (";
                        if (packType == "uint64_t") {
                            mixedPrecisionComputation << packType << "*)((" << elemTypeToCType(arg->elemType()) << "*)" << arg->name() <<" + offset);" << std::endl;
                        } else {
                            mixedPrecisionComputation << packType << "*)(" << arg->name() << "+ offset);" << std::endl;
                        }
                    }
                }

                mixedPrecisionComputation << indent(3) << "for (size_t ii = threadIdx.x; ii < nelem/packFactor; ii += blockDim.x) {"
                            << std::endl;
                
                //Declare pack variables
                for (auto stage : computationStages) {
                    mixedPrecisionComputation << indent(4) << vecTypeForStage[stage] << " " << stage->name() << " = MULTI<" << std::get<0>(allReduceKernelStructs[stage]).name<< "<T>" << ", T>()." <<  std::get<0>(allReduceKernelStructs[stage]).name << "(";
                
                    //FIXME: To the call to MULTI, we add all arguments other than 
                    //the result of Stage and the argument of AllReduce.
                    InputsVisitor inVisitor;
                    auto inputs = inVisitor.inputs(*stage);

                    for (auto iter = inputs.begin(); iter != inputs.end();) {
                        auto arg = *iter;
                        //If it is an argument to pipe then we emit "+ offset" too.
                        //We do not check for explicit Store locations right now to transfer data through register.
                        if (arg->type() == TensorNode) {
                            mixedPrecisionComputation << "*(" << arg->name() << "Pack + ii)";
                        } else if (arg == commCollOutput.get()) {
                            mixedPrecisionComputation << "*(" << commCollArg->name() << "Pack + ii)";
                        } else {
                            mixedPrecisionComputation << arg->name();
                        }

                        if (++iter != inputs.end())
                            mixedPrecisionComputation << ", ";
                    }

                    mixedPrecisionComputation << ");" << std::endl;
                    ASSERT(false, "FIX");
                    //FIX
                    // if (pipeline.explicitStoreLocations().count(stage) > 0) {
                    //     //If explicitly stored at a location then store it there 

                    //     // char storeCodeFmt[] = "if (((ii*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {\n"
                    //     //                 "    // Last incomplete word\n"
                    //     //                 "    storeAL(%sPack+offset, %s, nbytes & 0x7);\n"
                    //     //                 "} else {\n"
                    //     //                 "    storeAL(%sPack+offset, %s, sizeof(uint64_t));\n"
                    //     //                 "}\n";
                    //     std::string locName = pipeline.explicitStoreLocations().at(stage)->name();
                    //     // char storeCode[1024];
                    //     // sprintf(storeCode, storeCodeFmt, nameStr, stage->name().c_str(), nameStr, stage->name().c_str());

                    //     mixedPrecisionComputation << indent(4) << "*(" << locName << "Pack + ii) = " << stage->name() << ";" << std::endl;
                    // }
                }

                mixedPrecisionComputation << indent(3) << "}" << std::endl << indent(2) << "__syncthreads();" << std::endl
                            << indent(2) << "}" << std::endl;
      
                std::string allReduceH = readFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h");
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL_INSERT_MIXED_PREC_COMPUT), 
                                                RINGLL_INSERT_MIXED_PREC_COMPUT.size(), mixedPrecisionComputation.str());
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_MIXED_PREC_COMPUT), 
                                                RINGLL128_INSERT_MIXED_PREC_COMPUT.size(), mixedPrecisionComputation.str());
                allReduceH = allReduceH.replace(allReduceH.find(RINGLL128_INSERT_MIXED_PREC_COMPUT), 
                                                RINGLL128_INSERT_MIXED_PREC_COMPUT.size(), mixedPrecisionComputation.str());
                writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceH);
            }

            /**********DONE (ncclAllReduceRingLLKernel)**********/


            /**********Writing to ncclAllReduceRingKernel**********/
            const std::string INSERT_RINGSIMPLE_SEND = "/*RingSimple: {INSERT SEND}*/";
            const std::string INSERT_RINGSIMPLE_RECV_REDUCE_SEND = "/*RingSimple: {INSERT RECV_REDUCE_SEND}*/";
            const std::string INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND = "/*RingSimple: {INSERT DIRECT_RECV_REDUCE_COPY_SEND}*/";
            const std::string INSERT_RINGSIMPLE_DIRECT_RECV_COPY_SEND = "/*RingSimple: {INSERT DIRECT_RECV_COPY_SEND}*/";
            const std::string INSERT_RINGSIMPLE_DIRECT_RECV = "/*RingSimple: {INSERT DIRECT_RECV}*/";
            const std::string INSERT_RINGSIMPLE_ARGS = "/*RingSimple: {INSERT ARGS}*/";
            const std::string INSERT_GenericOp_PARAM_DEFAULT = "/*{INSERT GenericOp PARAM DEFAULT}*/";
            const std::string INSERT_directRecvReduceCopySend_ARGS = "/*{INSERT directRecvReduceCopySend ARGs}*/";
            const std::string INSERT_GenericOp_PARAM = "/*{INSERT GenericOp PARAM}*/";
            const std::string INSERT_GenericOp_ARGS = "/*{INSERT GenericOp ARGS}*/";
            const std::string INSERT_ReduceOrCopyMulti_PARAMS = "/*{INSERT ReduceOrCopyMulti PARAMS}*/";
            const std::string INSERT_ReduceOrCopyMulti_ARGS = "/*{INSERT ReduceOrCopyMulti ARGS}*/";
            const std::string INSERT_ReduceCopy128b_PARAMS = "/*{INSERT ReduceCopy128b PARAMS}*/";
            const std::string INSERT_ReduceCopy128b_ARGS = "/*{INSERT ReduceCopy128b ARGS}*/";
            const std::string INSERT_ReduceCopy128bMulti_COMPUTATION = "/*{INSERT ReduceCopy128bMulti COMPUTATION}*/";
            const std::string INSERT_ReduceCopy128bMulti_ALLGATHER_COMPUTATION = "/*{INSERT ReduceCopy128bMulti ALL GATHER COMPUTE}*/";
            const std::string INSERT_SEND_VAL = "/*{INSERT ReduceCopy128bMulti SEND VAL}*/";
            const std::string INSERT_RINGSIMPLE_OUTPUT = "/*{RingSimple: {INSERT OUTPUT}}*/";
            const std::string INSERT_RINGSIMPLE_REDUCTION_END_LOOP = "/*RingSimple: REDUCTION {END FOR LOOP FOR}*/";
            const std::string INSERT_RINGSIMPLE_REDUCTION_PER_GPU = "/*RingSimple: REDUCTION {PER-GPU REDUCTION}*/";
            const std::string INSERT_RINGSIMPLE_REDUCTION_TRANSFER = "/*RingSimple: REDUCTION {TRANSFER}*/";
            const std::string INSERT_RINGSIMPLE_REDUCTION_BEGIN_LOOP = "/*RingSimple: REDUCTION {BEGIN FOR LOOP FOR}*/";
            const std::string INSERT_RINGSIMPLE_REDUCTION_COMPUTATION = "/*RingSimple: REDUCTION {COMPUTATION}*/";
            const std::string INSERT_RINGSIMPLE_SHMEM_FOR_REDUCTION = "/*RingSimple: {INSERT SHARED MEMORY FOR REDUCTION}*/";

            std::string primsSend = "prims." + sendCall.str();
            std::string primsRecvReduceSend = "prims." + recvReduceSendCall.str();
            std::stringstream primsDirectRecvReduceCopySend;

            if (hasReduceTensor) {
                primsDirectRecvReduceCopySend << "prims.recvReduceCopy(";
                std::string argName = commCollArg->name();
                primsDirectRecvReduceCopySend << argName << " + offset, "
                                              << "(T*)" << argName << " + offset, ";
                // for (auto iter = argsOtherThanInputandOutput.begin(); iter != argsOtherThanInputandOutput.end(); ++iter) {
                //     auto arg = *iter;
                //     if (arg->type() == VariableNode) {
                //         primsDirectRecvReduceCopySend << arg->name();
                //     } else {
                //         primsDirectRecvReduceCopySend << arg->name() << " + offset";
                //     }

                //     primsDirectRecvReduceCopySend << ", ";
                // }

                primsDirectRecvReduceCopySend << "nelem);";
            } else {
                primsDirectRecvReduceCopySend << "prims.directRecvReduceCopySend(";
                primsDirectRecvReduceCopySend << commCollArg->name() << " + offset, ";
                primsDirectRecvReduceCopySend << (*outputs.begin())->name() << " + offset, ";
                for (auto iter = argsOtherThanInputandOutput.begin(); iter != argsOtherThanInputandOutput.end(); ++iter) {
                    auto arg = *iter;
                    if (arg->type() == VariableNode) {
                        primsDirectRecvReduceCopySend << arg->name();
                    } else {
                        primsDirectRecvReduceCopySend << arg->name() << " + offset";
                    }

                    primsDirectRecvReduceCopySend << ", ";
                }

                primsDirectRecvReduceCopySend << "offset, nelem);";
            }
        
            std::string primsDirectRecvCopySend;
            std::string primsDirectRecv; 

                primsDirectRecvCopySend = "prims.directR" + recvCopySendCall.str().substr(1, recvCopySendCall.str().size() - 1) +
                                                  "offset, nelem  * " + std::to_string(sizeFactor) + ");";
                primsDirectRecv = "prims.directR" + recvCall.str().substr(1, recvCall.str().size() - 1)  + 
                                           "offset, nelem * " + std::to_string(sizeFactor) + ");";
            

            std::string allReduceHRingSimple = readFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h");
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_SEND),
                                                                INSERT_RINGSIMPLE_SEND.size(), primsSend);
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_RECV_REDUCE_SEND),
                                                                INSERT_RINGSIMPLE_RECV_REDUCE_SEND.size(), primsRecvReduceSend);
            if (hasReduceTensor) {
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_REDUCTION_END_LOOP),
                                                                    INSERT_RINGSIMPLE_REDUCTION_END_LOOP.size(), 
                                                                    indent(1) +"}\n");
                std::stringstream computationLoop;
                computationLoop << indent(1) << "for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {"<<std::endl
                                    << indent(2) << "int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));" <<std::endl
                                    << indent(2) << "ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));" << std::endl
                                    << indent(2) << "ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));" << std::endl
                                    << indent(2) << "ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;" << std::endl
                                    << indent(2) << "ssize_t offset; int nelem; int chunk;" <<std::endl
                                    << indent(2) << "chunk = ring->devUserRanks[0];" << std::endl
                                    << indent(2) << "offset = chunkOffset + chunk * realChunkSize;" << std::endl
                                    << indent(2) << "nelem = min(realChunkSize, size-offset);" << std::endl
                                    << packVariablesDecl.str()<<std::endl;
                
                std::stringstream perGPUReductionCompRingSimple;
                perGPUReductionCompRingSimple << computationLoop.str()
                                    << indent(2) << "for (ssize_t offset = threadIdx.x; offset < nelem" << "/" << packFactor << "; offset += blockDim.x){" << std::endl
                                    << reduceTensorDAGComputation.str() 
                                    << indent(3) << "MULTI<FUNC, T>()." << reduceTensorAtomicOpName << "(&" << reduceTensorStage->name().c_str() << ", " << atomicOpInput << ");"<< std::endl
                                    << indent(2) << "}" << std::endl
                                    << indent(1) << "}" << std::endl;
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_REDUCTION_PER_GPU),
                                                                    INSERT_RINGSIMPLE_REDUCTION_PER_GPU.size(), 
                                                                    perGPUReductionCompRingSimple.str());
                
                std::string amongGPUReductionCodeStr = std::string(amongGPUReductionCode);
                replaceAllSubString(amongGPUReductionCodeStr, "LLprims", "prims");
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_REDUCTION_TRANSFER),
                                                                    INSERT_RINGSIMPLE_REDUCTION_TRANSFER.size(), 
                                                                    amongGPUReductionCodeStr);
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND),
                                                                    INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND.size(), primsDirectRecvReduceCopySend.str());                                                                          
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_REDUCTION_COMPUTATION),
                                                                    INSERT_RINGSIMPLE_REDUCTION_COMPUTATION.size(),
                                                                    computationLoop.str() + afterReductionComputation.str() +
                                                                    "prims.send(" + (*outputs.begin())->name() + " + offset, nelem);\n"); 
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_SHMEM_FOR_REDUCTION),
                                                                    INSERT_RINGSIMPLE_SHMEM_FOR_REDUCTION.size(),  "__shared__ T " + reduceTensorStage->name() + ";\n" + 
                                                "if (threadIdx.x == 0) " + reduceTensorStage->name() + " = 0;\n__syncthreads();\n");
            } else if (onlyCommCollStageIsMixedPrec) {
                std::string primsRecvReduceCopySend = "prims." + recvReduceCopySendCall.str() + " nelem);";
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND),
                                                                    INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND.size(), 
                                                                    primsRecvReduceCopySend + "\n" + 
                                                                    mixedPrecisionComputation.str() + "\n" +
                                                                    indent(2) + "prims." + outputSendCall.str());
            } else {
                allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND),
                                                                    INSERT_RINGSIMPLE_DIRECT_RECV_REDUCE_COPY_SEND.size(), primsDirectRecvReduceCopySend.str());
            }
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_DIRECT_RECV_COPY_SEND),
                                                                INSERT_RINGSIMPLE_DIRECT_RECV_COPY_SEND.size(), primsDirectRecvCopySend);
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_DIRECT_RECV),
                                                                INSERT_RINGSIMPLE_DIRECT_RECV.size(), primsDirectRecv);
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_ARGS),
                                                                INSERT_RINGSIMPLE_ARGS.size(), ringLLArgs.str());
            allReduceHRingSimple = allReduceHRingSimple.replace(allReduceHRingSimple.find(INSERT_RINGSIMPLE_OUTPUT),
                                                                INSERT_RINGSIMPLE_OUTPUT.size(), (*outputs.begin())->name());
            writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceHRingSimple);

            /*Write to primitives_computation.h*/
            std::stringstream genericOpParamDefaultVal;

            for (auto arg : argsOtherThanInputandOutput) {
                if (arg->type() == VariableNode)
                    genericOpParamDefaultVal << "(T)0, ";
                else
                    genericOpParamDefaultVal << "NULL, ";
            }

            std::stringstream directRecvReduceCopySendArgs;
            for (auto arg : argsOtherThanInputandOutput) {
                directRecvReduceCopySendArgs << printArgumentForNCCLPrims(arg) << ", ";
            }

            std::stringstream otherGenericOpParams;
            for (auto arg : argsOtherThanInputandOutput) {
                otherGenericOpParams << arg->name() << ", ";
            }

            std::stringstream reduceOrCopyMultiParams;
            for (auto arg : argsOtherThanInputandOutput) {
                reduceOrCopyMultiParams << ", " << arg->name();
            }

            std::string primitivesH = readFile(NCCL_SRC_PATH+"collectives/device/primitives_computation.h.in");
            primitivesH = replaceAllSubString(primitivesH, INSERT_GenericOp_PARAM_DEFAULT, 
                                              genericOpParamDefaultVal.str());
            primitivesH = replaceAllSubString(primitivesH, INSERT_directRecvReduceCopySend_ARGS, 
                                              directRecvReduceCopySendArgs.str());
            primitivesH = primitivesH.replace(primitivesH.find(INSERT_GenericOp_PARAM),
                                             INSERT_GenericOp_PARAM.size(), otherGenericOpParams.str());
            primitivesH = primitivesH.replace(primitivesH.find(INSERT_GenericOp_ARGS),
                                              INSERT_GenericOp_ARGS.size(), directRecvReduceCopySendArgs.str());
            primitivesH = replaceAllSubString(primitivesH, INSERT_ReduceOrCopyMulti_PARAMS,
                                              reduceOrCopyMultiParams.str());
            writeFile(NCCL_DST_PATH+"collectives/device/primitives_computation.h", primitivesH);

            /*Write to common_kernel.h*/
            //Write binOp<N> function to MULTI128
            const std::string INSERT_PACK128_ARGS_CONV = "/*{INSERT ARGS CONV TO PACK128}*/";
            const std::string INSERT_MULTI128_OPERATOR = "/*{INSERT MULTI128<T> operator()}*/";
            
            std::stringstream pack128ArgsConv;
            
            for (auto arg : argsOtherThanInputandOutput) {
                if (arg->type() == VariableNode)
                    continue;

                pack128ArgsConv << indent(2) << "Pack128* " << arg->name() << "Pack = " << "((Pack128*)(" << arg->name() << "+elemOffset))+offset;" << std::endl;
            }

            std::unordered_map<std::shared_ptr<StageImpl>, std::stringstream> MULTI128OperatorForStage;

            for (auto iter : allReduceKernelStructs) {
                MULTI128OperatorForStage[iter.first] = std::stringstream();

                MULTI128OperatorForStage[iter.first] << "__device__ void " <<  std::get<0>(iter.second).name << "(";
                
                InputsVisitor inVisitor;
                auto inputs = inVisitor.inputs(*iter.first);
                
                //All inputs to a stage are input to MULTI128 function
                for (auto iterArg = inputs.begin(); iterArg != inputs.end(); ++iterArg) {
                    auto arg = *iterArg;
                    if (arg->type() == VariableNode) {
                        MULTI128OperatorForStage[iter.first] << "const " << elemTypeToCType(arg->elemType()) << " " << arg->name();
                    } else {
                        MULTI128OperatorForStage[iter.first] << "Pack128&" << " " << arg->name();
                    }

                    MULTI128OperatorForStage[iter.first] << ", ";
                }

                //Add output as argument
                MULTI128OperatorForStage[iter.first] << "Pack128&" << " " << iter.first->name() ;
                MULTI128OperatorForStage[iter.first] << ") {" << std::endl;

                ASSERT(outputs.size() == 1, "Only one output is supported right now.");

                MULTI128OperatorForStage[iter.first] << indent(2) << iter.first->name() << ".x = " << "MULTI<FUNC, T>()." << std::get<0>(iter.second).name << "(";
                for (auto iterArg = inputs.begin(); iterArg != inputs.end();) {
                    auto arg = *iterArg;
                    if (arg->type() == VariableNode) {
                        MULTI128OperatorForStage[iter.first] << arg->name();
                    } else {
                        MULTI128OperatorForStage[iter.first] << arg->name() << ".x";
                    }

                    if (++iterArg != inputs.end()) {
                        MULTI128OperatorForStage[iter.first] << ", ";
                    }
                }
                MULTI128OperatorForStage[iter.first] << ");" << std::endl;
                MULTI128OperatorForStage[iter.first] << indent(2) << iter.first->name() << ".y = " << "MULTI<FUNC, T>()." << std::get<0>(iter.second).name << "(";
                for (auto iterArg = inputs.begin(); iterArg != inputs.end();) {
                    auto arg = *iterArg;
                    if (arg->type() == VariableNode) {
                        MULTI128OperatorForStage[iter.first] << arg->name();
                    } else {
                        MULTI128OperatorForStage[iter.first] << arg->name() << ".y";
                    }

                    if (++iterArg != inputs.end()) {
                        MULTI128OperatorForStage[iter.first] << ", ";
                    }
                }
                MULTI128OperatorForStage[iter.first] << ");" << std::endl;
                //}

                MULTI128OperatorForStage[iter.first] << indent(1) << "}";
            }
            
            //Update parameters of ReduceOrCopyMulti and ReduceCopyMulti
            std::stringstream reduceOrCopyMultiARGS;
            for (auto arg : argsOtherThanInputandOutput) {
                reduceOrCopyMultiARGS << ", " << printArgumentForNCCLPrims(arg);
            }

            std::stringstream reduceCopy128bMultiComputation;
            std::stringstream reduceCopy128bMultiAllGatherComputaiton;

            if (not onlyCommCollStageIsMixedPrec and not hasReduceTensor) {
                //For all args other than the commCollStage's arg, fetch their value in a Pack128
                for (auto arg : argsOtherThanInputandOutput) {
                    if (arg->type() == VariableNode)
                        continue;
                    if (outputs.find(arg) != outputs.end())
                    //If an output then do not declare it
                        continue;
                    reduceCopy128bMultiComputation << indent(2) << "Pack128 " << arg->name() << "PackVal;" << std::endl;
                    reduceCopy128bMultiComputation << indent(2) << "Fetch128(" << arg->name() << "PackVal, " << arg->name() << "Pack + u*WARP_SIZE);" << std::endl;
                }

                for (auto stage : computationStages) {
                    std::stringstream computation;

                    computation << indent(2) << "Pack128 " << stage->name() << "Pack;" << std::endl;
                    
                    computation << indent(2) << "MULTI128<" << std::get<0>(allReduceKernelStructs[stage]).name << "<T>, T>()." <<  std::get<0>(allReduceKernelStructs[stage]).name << "(";

                    InputsVisitor inVisitor;
                    auto inputs = stage->usedExprs();

                    bool hasAllGatherStageAsArg = false;

                    for (auto iter = inputs.begin(); iter != inputs.end();) {
                        auto arg = *iter;

                        if (arg == commCollArg || arg == commCollOutput) {
                            computation << "finalVal";
                        } else if (outputs.find(arg) != outputs.end()) {
                            computation << "readVal";
                        } else if (arg->type() == VariableNode) {
                            computation << arg->name();
                        } else {
                            if (fusedToAllReduce and arg == allGatherStage) {
                                //If current arg is AllGather's input then 
                                //generate finalVal
                                computation << "%s";
                                hasAllGatherStageAsArg = true;
                            } else if (argsOtherThanInputandOutput.find(arg) != argsOtherThanInputandOutput.end()) {
                                computation << arg->name() << "PackVal";
                            } else {
                                computation << arg->name() << "Pack";
                            }
                        }

                        if (++iter != gpuKernelArgs.end()) {
                            computation << ", ";
                        }
                    }

                    if (outputStages.find(stage) == outputStages.end()) {
                        if (fusedToAllReduce and stage == AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()) {
                            computation << "%l);" << std::endl;
                        } else 
                            computation << stage->name () << "Pack);" << std::endl;
                    } else {
                        computation << "finalVal);" << std::endl;
                    }

                    if (fusedToAllReduce) {
                        //If ReduceScatter and AllGather are fused to AllReduce then do not generate computation for the output.
                        bool isStageAfterAllGather = std::find(stagesAfterAllGather.begin(), stagesAfterAllGather.end(), stage) != stagesAfterAllGather.end();
                        if (outputStages.count(stage) > 0 || isStageAfterAllGather) {
                            std::string ss = computation.str();
                            replaceAllSubString(ss, "%s", "finalVal");
                            reduceCopy128bMultiAllGatherComputaiton << ss;
                        }
                        
                        std::string ss = computation.str();
                        if (hasAllGatherStageAsArg) { 
                            replaceAllSubString(ss, "%s", AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()->name() + "Pack");
                        }

                        if (stage == AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()) {
                            replaceAllSubString(ss, "%l", stage->name() + "Pack");
                        }
                        
                        reduceCopy128bMultiComputation << ss;
                    } else {
                        reduceCopy128bMultiComputation << computation.str();
                    }
                }
            }

            std::string MULTI128Operator = "";
            for (auto stage : computationStages) {
                MULTI128Operator += MULTI128OperatorForStage[stage].str() + "\n";
            }
            //We are rewriting this file after updating for RingLL
            commonKernelH = readFile(NCCL_DST_PATH+"collectives/device/common_kernel.h");
            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_MULTI128_OPERATOR),
                                                  INSERT_MULTI128_OPERATOR.size(), MULTI128Operator);
            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_ReduceOrCopyMulti_ARGS),
                                                  INSERT_ReduceOrCopyMulti_ARGS.size(), reduceOrCopyMultiARGS.str());
            commonKernelH = replaceAllSubString(commonKernelH, INSERT_ReduceCopy128b_PARAMS, reduceOrCopyMultiParams.str());
            commonKernelH = replaceAllSubString(commonKernelH, INSERT_ReduceCopy128b_ARGS, reduceOrCopyMultiARGS.str());
            std::string sendVal;
            if (fusedToAllReduce) {
                sendVal = AstNodeImpl::asAllGatherImpl(allGatherStage->definition())->arg()->name() + "Pack";
            } else {
                sendVal = "finalVal";
            }
            commonKernelH = replaceAllSubString(commonKernelH, INSERT_SEND_VAL, sendVal);
            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_ReduceCopy128bMulti_COMPUTATION),
                                                  INSERT_ReduceCopy128bMulti_COMPUTATION.size(), reduceCopy128bMultiComputation.str());
            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_PACK128_ARGS_CONV),
                                                  INSERT_PACK128_ARGS_CONV.size(), pack128ArgsConv.str());
            commonKernelH = commonKernelH.replace(commonKernelH.find(INSERT_ReduceCopy128bMulti_ALLGATHER_COMPUTATION),
                                                  INSERT_ReduceCopy128bMulti_ALLGATHER_COMPUTATION.size(), 
                                                  reduceCopy128bMultiAllGatherComputaiton.str());
            writeFile(NCCL_DST_PATH+"collectives/device/common_kernel.h", commonKernelH);
            /**********DONE (ncclAllReduceRingKernel)**********/

            /**********ncclAllReduceTreeLLKernel**********/
            const std::string INSERT_TREELL_ARGS = "/*TREELL: {INSERT ARGS}*/";
            const std::string INSERT_TREELL_RECV_REDUCE_COPY = "/*TREELL: {INSERT recvReduceCopy2}*/";
            const std::string INSERT_TREELL_SEND = "/*TREELL: {INSERT send}*/";
            const std::string INSERT_TREELL_RECV_REDUCE_SEND = "/*TREELL: {INSERT recvReduceSend}*/";
            const std::string INSERT_TREELL_SEND_OUTPUT = "/*TREELL: {INSERT send output}*/";
            const std::string INSERT_TREELL_RECV_OUTPUT = "/*TREELL: {INSERT recv output}*/";
            const std::string INSERT_TREELL_RECV_COPY_SEND_OUTPUT = "/*TREELL: {INSERT recvCopySend output}*/";
            const std::string INSERT_LLPRIMS_RECV_REDUCE_COPY_ARGS = "/*{INSERT recvReduceCopy ARGS}*/";
            const std::string INSERT_TREELL_REDUCTION_REDUCTION_COMPUTATION = "/*TREELL: REDUCTION {REDUCTION COMPUTATION}*/";
            const std::string INSERT_TREELL_REDUCTION_GLOBAL_MEM = "/*TREELL: REDUCTION {GLOBAL MEM REDUCTION}*/";
            const std::string INSERT_TREELL_REDUCTION_COMPUTATION = "/*TREELL: REDUCTION {COMPUTATION}*/";
            const std::string INSERT_TREELL_REDUCTION_SHMEM_DECL = "/*TREELL: {REDUCTION SHMEM}*/";

            std::stringstream treeSendOutput;
            std::stringstream treeRecvOutput;
            std::stringstream treeRecvCopySendOutput;

            treeSendOutput << "send(";
            if (fusedToAllReduce) {
                treeRecvOutput << "recvAllGatherCompute(";
                treeRecvCopySendOutput << "recvCopySendAllGatherCompute(";
            } else {
                treeRecvOutput << "recv(";
                treeRecvCopySendOutput << "recvCopySend(";
            }

            if (onlyCommCollStageIsMixedPrec) {
                std::string outputCTypeStr = elemTypeToCType((*outputs.begin())->elemType());
                treeSendOutput << "(T*)((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << " + offset), nelem * " << sizeFactor << ");" << std::endl;
                treeRecvOutput << "(T*)((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << " + offset), nelem * " << sizeFactor<< ");" << std::endl;
                treeRecvCopySendOutput << "(T*)((" << outputCTypeStr << "*)" << (*outputs.begin())->name() << " + offset), nelem * " << sizeFactor << ");" << std::endl;
            } else {
                treeSendOutput << (*outputs.begin())->name() << " + offset, nelem);" << std::endl;
                treeRecvOutput << (*outputs.begin())->name() << " + offset, nelem);" << std::endl;
                treeRecvCopySendOutput << (*outputs.begin())->name() << " + offset, nelem);" << std::endl;
            }

            std::string allReduceHTreeLL = readFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h");
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_SEND),
                                                        INSERT_TREELL_SEND.size(), "LLprims." + sendCall.str());
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_RECV_REDUCE_SEND),
                                                        INSERT_TREELL_RECV_REDUCE_SEND.size(), "LLprims." + recvReduceSendCall.str());
            std::string recvReduceCopySendCallReplacement = "LLprims." + recvReduceCopySendCall.str() + "nelem);";
            replaceAllSubString(recvReduceCopySendCallReplacement, "recvReduceCopySend", "recvReduceCopy2");
            if (onlyCommCollStageIsMixedPrec) {
                recvReduceCopySendCallReplacement += "\n" + mixedPrecisionComputation.str();
            }

            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_RECV_REDUCE_COPY),
                                                        INSERT_TREELL_RECV_REDUCE_COPY.size(), recvReduceCopySendCallReplacement);
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_SEND_OUTPUT),
                                                         INSERT_TREELL_SEND_OUTPUT.size(), "LLprims." + treeSendOutput.str());
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_RECV_OUTPUT),
                                                        INSERT_TREELL_RECV_OUTPUT.size(), "LLprims." + treeRecvOutput.str());
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_ARGS),
                                                        INSERT_TREELL_ARGS.size(), ringLLArgs.str());
            allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_RECV_COPY_SEND_OUTPUT),
                                                        INSERT_TREELL_RECV_COPY_SEND_OUTPUT.size(), "LLprims." + treeRecvCopySendOutput.str());
            if (hasReduceTensor) {
                std::stringstream treeLLPerGPUReductionComp;
                std::stringstream computationLoop;        
        
                computationLoop << indent(1) << "for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {"<<std::endl
                                << indent(2) << "ssize_t offset = gridOffset + bid*chunkSize;" << std::endl
                                << indent(2) << "int nelem = min(chunkSize, size-offset);" << std::endl
                                << indent(1) << packVariablesDecl.str()<<std::endl;
                treeLLPerGPUReductionComp << computationLoop.str()
                                          << indent(2) << "for (ssize_t offset = threadIdx.x; offset < nelem" << "/" << packFactor << "; offset += blockDim.x){" << std::endl
                                        << reduceTensorDAGComputation.str() 
                                        << indent(3) << "MULTI<FUNC, T>()." << reduceTensorAtomicOpName << "(&" << reduceTensorStage->name().c_str() << ", " << atomicOpInput << ");"<< std::endl
                                        << indent(2) << "}" << std::endl
                                        << indent(1) << "}" << std::endl;

                allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_REDUCTION_REDUCTION_COMPUTATION),
                                                            INSERT_TREELL_REDUCTION_REDUCTION_COMPUTATION.size(), treeLLPerGPUReductionComp.str());
                std::stringstream globalMemReduction;
                globalMemReduction << "if(threadIdx.x + blockDim.x*blockIdx.x == 0) args->comm->reduced" << reduceTensorStage->name() << " = 0;" << std::endl
                                   << "cooperative_groups::grid_group __grid_group = cooperative_groups::this_grid();" << std::endl
                                   << "__grid_group.sync();" << std::endl
                                   << "if(threadIdx.x == 0) {" << std::endl
                                   << "::atomicAdd(&args->comm->reduced" << reduceTensorStage->name() << ", (float)" << reduceTensorStage->name() << ");" << std::endl
                                   << "}" << std::endl
                                   << "__grid_group.sync();" << std::endl
                                   << reduceTensorStage->name() << " = args->comm->reduced" << reduceTensorStage->name() << ";" << std::endl;
                allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_REDUCTION_GLOBAL_MEM),
                                                            INSERT_TREELL_REDUCTION_GLOBAL_MEM.size(), globalMemReduction.str());
                allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_REDUCTION_COMPUTATION),
                                                            INSERT_TREELL_REDUCTION_COMPUTATION.size(), computationLoop.str() + afterReductionComputation.str() + "}");

                allReduceHTreeLL = allReduceHTreeLL.replace(allReduceHTreeLL.find(INSERT_TREELL_REDUCTION_SHMEM_DECL),
                                                            INSERT_TREELL_REDUCTION_SHMEM_DECL.size(),
                                                            "__shared__ T " + reduceTensorStage->name() + ";\n" + 
                                                            "if (threadIdx.x == 0) " + reduceTensorStage->name() + " = 0;\n__syncthreads();\n");
            }
            writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceHTreeLL);

            //Update prims_ll_computation.h.in
            primsLLH = readFile(NCCL_DST_PATH + "collectives/device/prims_ll_computation.h");
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_LLPRIMS_RECV_REDUCE_COPY_ARGS),
                                        INSERT_LLPRIMS_RECV_REDUCE_COPY_ARGS.size(), recvReduceCopySendDecl.str());
            primsLLH = primsLLH.replace(primsLLH.find(INSERT_LLGenericOp2_CALL_PARAMS),
                                        INSERT_LLGenericOp2_CALL_PARAMS.size(), LLGenericOp2CallParams.str());
            writeFile(NCCL_DST_PATH + "collectives/device/prims_ll_computation.h", primsLLH);
            /**********DONE (ncclAllReduceTreeLLKernel)**********/

            /***************ncclAllReduceTreeKernel**************/
            
            const std::string INSERT_TREESIMPLE_ARGS = "/*TREESimple: {INSERT ARGS}*/";
            const std::string INSERT_TREESIMPLE_RECV_REDUCE_COPY = "/*TREESimple: {INSERT RECV_REDUCE_COPY}*/";
            const std::string INSERT_TREESIMPLE_SEND = "/*TREESimple: {INSERT SEND}*/";
            const std::string INSERT_TREESIMPLE_RECV_REDUCE_SEND = "/*TREESimple: {INSERT RECV_REDUCE_SEND}*/";
            const std::string INSERT_TREESIMPLE_SEND_OUTPUT = "/*TREESimple: {INSERT SEND OUTPUT}*/";
            const std::string INSERT_TREESIMPLE_RECV_OUTPUT = "/*TREESimple: {INSERT RECV OUTPUT}*/";
            const std::string INSERT_TREESIMPLE_RECV_COPY_SEND = "/*TREESimple: {INSERT RECV_COPY_SEND}*/";
            const std::string INSERT_RECV_REDUCE_COPY_ARGS = "/*{INSERT recvReduceCopy ARGs}*/";

            std::stringstream recvReduceCopyCall;

            if (onlyCommCollStageIsMixedPrec) {
                std::string argName = commCollArg->name();
                recvReduceCopyCall << "recvReduceCopy(";
                recvReduceCopyCall << argName << " + offset, (T*)" << argName << " + offset, nelem);";
            } else {
                recvReduceCopyCall << "recvReduceCopy2(";
                recvReduceCopyCall << commCollArg->name() << " + offset, ";
                recvReduceCopyCall << (*outputs.begin())->name() << " + offset, ";

                for (auto arg : argsOtherThanInputandOutput) {
                    if (arg->type() == VariableNode) {
                        recvReduceCopyCall << arg->name();
                    } else {
                        recvReduceCopyCall << arg->name() << " + offset";
                    }

                    recvReduceCopyCall << ", ";
                }

                recvReduceCopyCall << "nelem);";
            }

            if (onlyCommCollStageIsMixedPrec) {
                recvReduceCopySendCallReplacement = "prims." + recvReduceCopySendCall.str() + "nelem);";
                replaceAllSubString(recvReduceCopySendCallReplacement, "recvReduceCopySend", "recvReduceCopy2");
                recvReduceCopySendCallReplacement += "\n" + mixedPrecisionComputation.str();
            } else {
                recvReduceCopySendCallReplacement = "prims." + recvReduceCopyCall.str();
            }

            std::string allReduceHTreeSimple = readFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h");
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_SEND),
                                                                INSERT_TREESIMPLE_SEND.size(), "prims." + sendCall.str());
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_RECV_REDUCE_SEND),
                                                                INSERT_TREESIMPLE_RECV_REDUCE_SEND.size(), "prims." + recvReduceSendCall.str());
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_RECV_REDUCE_COPY),
                                                                INSERT_TREESIMPLE_RECV_REDUCE_COPY.size(), recvReduceCopySendCallReplacement);
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_SEND_OUTPUT),
                                                                INSERT_TREESIMPLE_SEND_OUTPUT.size(), "prims." + treeSendOutput.str());
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_RECV_OUTPUT),
                                                                INSERT_TREESIMPLE_RECV_OUTPUT.size(), "prims." + treeRecvOutput.str());
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_ARGS),
                                                                INSERT_TREESIMPLE_ARGS.size(), ringLLArgs.str());
            allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(INSERT_TREESIMPLE_RECV_COPY_SEND),
                                                                INSERT_TREESIMPLE_RECV_COPY_SEND.size(), "prims." + treeRecvCopySendOutput.str());
            if (onlyCommCollStageIsMixedPrec) {
                std::string TREESIMPLE_StepSize_MP_FACTOR = "/*{TREESimple: {stepSize MIXED PRECISION FACTOR}*/";
                allReduceHTreeSimple = allReduceHTreeSimple.replace(allReduceHTreeSimple.find(TREESIMPLE_StepSize_MP_FACTOR),
                                                                    TREESIMPLE_StepSize_MP_FACTOR.size(), "*2");
            }
            writeFile(NCCL_DST_PATH+"collectives/device/all_reduce_computation.h", allReduceHTreeSimple);

            std::stringstream primsRecvReduceCopy;
            for (auto arg : argsOtherThanInputandOutput) {
                primsRecvReduceCopy << printArgumentForNCCLPrims(arg) << ", ";
            }

            primitivesH = readFile(NCCL_DST_PATH + "collectives/device/primitives_computation.h");
            primitivesH = primitivesH.replace(primitivesH.find(INSERT_GenericOp_PARAM),
                                              INSERT_GenericOp_PARAM.size(), otherGenericOpParams.str());
            primitivesH = primitivesH.replace(primitivesH.find(INSERT_RECV_REDUCE_COPY_ARGS),
                                              INSERT_RECV_REDUCE_COPY_ARGS.size(), primsRecvReduceCopy.str());
            writeFile(NCCL_DST_PATH + "collectives/device/primitives_computation.h", primitivesH);

            break;
        }

        default:
            ASSERT(false, "Not implemented for comm coll type " << AstNodeTypeToStr(commCollType));
    }

    switch (commCollType) {
        case AllReduceNode: {
            
            break;
        }
        
        default:
            ASSERT(false, "Add support for comm coll " << commCollType);
    }

    return ncclFuncCallStr;
}

std::string genCUDAFuncCall(std::shared_ptr<StageImpl> outStage, CFunc& cfunc, std::string streamArg, int indentLevel) 
{
    // ASSERT(outStage->dims() == 1, "Only 1 dimension tensors supported");
    ASSERT(cfunc.isCUDA, "Function is not a CUDA kernel");
    std::stringstream totalThreads;
    static int kernelCall = 0;
    std::string totalThreadsVar = "totalThreads_" + std::to_string(kernelCall);
    auto stageDef = outStage->definition() ;

    if (stageDef->type() == UpdateNode)
        stageDef = AstNodeImpl::asUpdateImpl(stageDef)->update();
    
    std::string numThreadsVar = "numThreads_"+ std::to_string(kernelCall);
    std::string numThreadBlocksVar = "numThreadBlocks_"+std::to_string(kernelCall);
    
    std::stringstream numThreads;
    std::stringstream numThreadBlocks;
    std::stringstream call;
    if (stageDef->type() == MatMulNode) {
        call << indent(indentLevel) << cfunc.name << "(";
        int ii = 0;
        for (auto iter : cfunc.arguments) {
            call << iter->name();
            if (ii != cfunc.arguments.size() - 1)
                call << ", ";
            ii++;
        }

        if (stageDef->type() == MatMulNode)
            call << ", " << cublasHandleVar;
        call << ", " << commSizeArg << ", " << rankVar << ")";

    } else if (cfunc.useCooperativeGrid == false) {
        if (stageDef->type() == BinaryPointwiseOpNode || stageDef->type() == DropoutNode) {
            totalThreads << "size_t " << totalThreadsVar << " = (size_t)" << genNumElem(outStage) << ";";
        } else if (stageDef->type() == ReduceTensorNode) {
            totalThreads << "size_t " << totalThreadsVar << " = (size_t)" << genNumElem(AstNodeImpl::asReduceTensorImpl(stageDef)->arg()) << ";";
        } else if (stageDef->type() == NormNode) {
            totalThreads << "size_t " << totalThreadsVar << " = (size_t)" << genNumElem(AstNodeImpl::asNormImpl(stageDef)->arg()) << ";";
        } else {
            ASSERT(false, "Not implemented for node type\n");
        }

        numThreads << "size_t " << numThreadsVar << " = (size_t)min(" << totalThreadsVar << ", 256UL);";
        numThreadBlocks << "size_t " << numThreadBlocksVar << " = DIVUP(" << totalThreadsVar << ", " << numThreadsVar << ");";

        call << indent(indentLevel) << totalThreads.str() << std::endl << indent(indentLevel) << numThreads.str() << std::endl << indent(indentLevel) << numThreadBlocks.str() << std::endl;
        call << indent(indentLevel) << cfunc.name << "<<<" << numThreadBlocksVar << ", " << numThreadsVar << ", "<< 0 << ", " << streamArg <<">>>(";
        int ii = 0;
        for (auto iter : cfunc.arguments) {
            call << iter->name();
            if (ii != cfunc.arguments.size() - 1)
                call << ", ";
            ii++;
        }

        if (stageDef->type() == MatMulNode)
            call << ", " << cublasHandleVar;
        call << ", " << commSizeArg << ", " << rankVar << ")";
    } else {
        numThreads << "dim3 " << numThreadsVar << " = {256, 1, 1};" << std::endl;
        numThreadBlocks << "dim3 " << numThreadBlocksVar << " = {1, 1, 1};" << std::endl;

        std::string kernelArgsVar = "args_"+std::to_string(kernelCall);
        std::string maxActiveBlocks = printCudaOccupancyMaxActiveBlocksPerMultiprocessor(numThreadBlocksVar+".x", cfunc.name, numThreadsVar+".x", 0);
        
        call << indent(indentLevel) << numThreads.str();
        call << indent(indentLevel) << numThreadBlocks.str();
        call << indent(indentLevel) << cudaCheck(maxActiveBlocks) << std::endl;
        call << indent(indentLevel) << numThreadBlocksVar << ".x = 80 * " << numThreadBlocksVar << ".x;" << std::endl;
        call << indent(indentLevel) << "void* " << kernelArgsVar << "[] = {";
        int ii = 0;
        for (auto iter : cfunc.arguments) {
            call << "&" << iter->name();
            if (ii != cfunc.arguments.size() - 1)
                call << ", ";
            ii++;
        }
        call << ", &" << commSizeArg << ", &" << rankVar << "};" << std::endl;
        call << indent(indentLevel) << cudaCheck(printCudaLaunchCooperativeKernel(cfunc.name, numThreadBlocksVar, numThreadsVar, kernelArgsVar, 0, streamArg)) << std::endl;
    }

    /**
     * 
  dim3 threads = {256, 1,1};
  dim3 blocks = {1,1,1};
  int blockspersm  = 0;
  CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blockspersm, (void*)binOpFunc0, threads.x, 0));
    blocks.x = 80 * blockspersm;
  void* args[] = {
    &N, &lr, &beta1, &beta2, &gamma, &w, &g, &m, &v, &S5, &S6, &S7, &S8, &comm_size, &rank
  };
  CUDACHECK(cudaLaunchCooperativeKernel((void*)binOpFunc0, blocks, threads, args, 0, stream));
     */
    
    kernelCall++;
    return call.str();
}

struct IntermediateStage {
    std::shared_ptr<StageImpl> stageImpl;
    bool requiresAlloc;
};

bool isStageAnIntermediate(std::vector<IntermediateStage>& intermStages, std::shared_ptr<ExpressionImpl> stage)
{
    if (stage->type() != StageNode)
        return false;

    for (auto interm : intermStages) {
        if (interm.stageImpl == AstNodeImpl::asStageImpl(stage)) {
            return true;
        }
    }

    return false;
}

void printIntermdiatesCUDAAlloc(std::vector<IntermediateStage>& intermediates, std::string& declCode, 
                                std::string& allocCode, std::string& freeCode, int indentLevel)
{
    /*Intermediate CUDA Arrays are allocated and freed*/
    //TODO: We do not need to allocate everything because we can reuse
    //intermediates storage between stages.
    std::stringstream declCodeStream, allocCodeStream, freeCodeStream;
    for (auto it : intermediates) {
        declCodeStream << indent(indentLevel) << printDeclaration(it.stageImpl) << std::endl;
        if (it.requiresAlloc) {
            allocCodeStream << indent(indentLevel) << printCUDAMalloc(it.stageImpl) << std::endl;
            freeCodeStream << indent(indentLevel) << printCUDAFree(it.stageImpl) << std::endl;
        }
    }

    declCode = declCodeStream.str();
    allocCode = allocCodeStream.str();
    freeCode = freeCodeStream.str();
}

void ACCCDSLImpl::NCCLCodegen::codegen(std::vector<CodeGenVarBounds> varBounds)
{
    /*Generate the reference function*/
    /*Print Function declaration*/
    const std::string commArg = "comm";
    const std::string streamArg = "stream";
    const std::string ncclCommTy = "ncclComm_t";
    const std::string streamTy = "cudaStream_t";

    std::vector<ArrayDecl> arrayDecls;
    std::stringstream funcBody;
    std::unordered_map<std::shared_ptr<StageImpl>, ArrayDecl> stageToArrayDecl;
    std::vector<CFunc> subFunctions;
    std::vector<IntermediateStage> intermediateStages;
    std::unordered_map<PipelineStage*, std::pair<std::string, std::string>> psToNameAndTimeVar;
    pipeline_.setAllStageStoreLoc();

    const std::string startEvent = "start" + pipeline_.name();
    const std::string stopEvent = "stop" + pipeline_.name();
    const std::string elapsedTimeVar = "elapsedTime";

    funcBody << indent(1) << "cudaEvent_t " << startEvent << ", "  << stopEvent << ";" << std::endl
             //Declare time variable
             << indent(1) << "float " << elapsedTimeVar << ";" << std::endl
             << indent(1) << printEventCreate(startEvent) << std::endl
             << indent(1) << printEventCreate(stopEvent) << std::endl << std::endl;
    
    bool hasACommCollStage = false;
    bool useCUBLAS = false;

    //Generate code in a topological sort manner
    for (auto pipelineStage : pipeline_.topoOrder()) {
        std::string stageName;
        std::string pipelineStageName = "";

        for (auto outStage : pipelineStage->stages()) {
            if (pipeline_.explicitStoreLocations().count(outStage) == 1) {
                stageName = pipeline_.explicitStoreLocations().at(outStage)->name();
            } else {
                stageName = outStage->name();
                //All stages are stored in CUDA
                if (pipelineStage->getStorageLocation(outStage) == 
                    StageStoreLocation::Memory) {
                    ArrayDecl stageArray = {stageName, outStage->dimSizes(), true};
                    stageToArrayDecl[outStage] = stageArray;
                    arrayDecls.push_back(stageArray);
                }
                pipelineStageName += "_" + stageName;
            }
        }
        
        funcBody << indent(1) << printEventRecord(startEvent, "stream") << std::endl;

        /* We characterize each Stage into following cases:
        *  1.   A stage is initialized with a Communication Collective.
        *  2.   A stage is initialized with some binary pointwise expression that can be
        *       converted to a cublas call.(TODO)
        *  3.   stage(or stages if fused) is initialized with some binary pointwise for which a CUDA kernel
        *       needs to be generated.
        * */
        if (pipelineStage->stages().size() == 1) {
            //If there is only one stage
            bool inPlace = false;

            for (auto outStage : pipelineStage->stages()) {
                std::shared_ptr<ExpressionImpl> stageDef = outStage->definition();                
                if (stageDef->type() == UpdateNode)
                    stageDef = AstNodeImpl::asUpdateImpl(stageDef)->update();

                switch (stageDef->type()) {
                    //Definition of collective communications contains only stage or tensor.
                    case AllReduceNode: {
                        std::shared_ptr<AllReduceImpl> allReduceColl = AstNodeImpl::asAllReduceImpl(stageDef);
                        funcBody << indent(1) << "ncclAllReduce(" << allReduceColl->arg()->name() << ", " << stageName << ", " << 
                            genNumElem(allReduceColl->arg()) << ", " << elemTypeToNCCLType(allReduceColl->arg()->elemType()) << "," << 
                            redOpToNCCLReduceOp(allReduceColl->reduceOp()) << ", " << commArg << ", " << streamArg << ");" << std::endl;
                        pipelineStageName = "AllReduce";
                        inPlace = true;
                        break;
                    }
                    case AllGatherNode: {
                        std::shared_ptr<AllGatherImpl> allGatherColl = AstNodeImpl::asAllGatherImpl(stageDef);
                        funcBody << indent(1) << "ncclAllGather(" << allGatherColl->arg()->name() << ", " << stageName << ", " << 
                            genNumElem(allGatherColl->arg()) << ", " << elemTypeToNCCLType(allGatherColl->arg()->elemType()) << 
                            ", " << commArg << ", " << streamArg << ");" << std::endl;
                        pipelineStageName = "AllGather";
                        break;
                    }
                    case ReduceScatterNode: {
                        std::shared_ptr<ReduceScatterImpl> reduceScatterColl = AstNodeImpl::asReduceScatterImpl(stageDef);
                        funcBody << indent(1) << "ncclReduceScatter(" << reduceScatterColl->arg()->name() << ", " << stageName << ", " << 
                            genNumElem(reduceScatterColl) << ", " << elemTypeToNCCLType(reduceScatterColl->arg()->elemType()) << 
                            ", " << redOpToNCCLReduceOp(reduceScatterColl->reduceOp()) << ", " <<
                            commArg << ", " << streamArg << ");" << std::endl;
                        pipelineStageName = "ReduceScatter";
                        break;
                    }
                    case ReduceNode: {
                        std::shared_ptr<ReduceImpl> reduceColl = AstNodeImpl::asReduceImpl(stageDef);
                        funcBody << indent(1) << "ncclReduce(" << reduceColl->arg()->name() << ", " << stageName << ", " << 
                            genNumElem(reduceColl->arg()) << ", " << elemTypeToNCCLType(reduceColl->arg()->elemType()) << 
                            redOpToNCCLReduceOp(reduceColl->reduceOp()) << ", " << reduceColl->root() << ", " <<
                            commArg << ", " << streamArg << ");" << std::endl;
                        pipelineStageName = "ReduceNode";
                        break;
                    }
                    case BroadcastNode: {
                        std::shared_ptr<BroadcastImpl> broadcastColl = AstNodeImpl::asBroadcastImpl(stageDef);
                        funcBody << indent(1) << "ncclBroadcast(" << broadcastColl->arg()->name() << ", " << stageName << ", " << 
                            genNumElem(broadcastColl->arg()) << ", " << elemTypeToNCCLType(broadcastColl->arg()->elemType()) << 
                            ", " << broadcastColl->root() << ", " <<
                            commArg << ", " << streamArg << ");" << std::endl;
                        pipelineStageName = "Broadcast";
                        break;
                    }
                    case BinaryPointwiseOpNode: {
                        CFunc cfunc = generateBinOpCodeCUDA(pipeline_, outStage, AstNodeImpl::asBinaryPointwiseOp(stageDef));
                        subFunctions.push_back(cfunc);
                        funcBody << genCUDAFuncCall(outStage, cfunc, streamArg, 1)<<";"<<std::endl;
                        pipelineStageName = cfunc.name;
                        break;
                    }
                    case ScatterNode: {
                        //A Scatter Node 
                        std::shared_ptr<ScatterImpl> scatter = AstNodeImpl::asScatterImpl(stageDef);
                        printf("FIXME: %s:%d\n", __FILE__, __LINE__);
                        funcBody << indent(1) << stageName << " = " << scatter->arg()->name() << " + " << 
                            scatter->size(0) << "/" << "FIX"/*scatter->numGPUs()*/ << " * " << rankString << std::endl;
                        break;
                    }
                    case ReduceTensorNode: {
                        CFunc cfunc = generateReduceCUDA(pipeline_, outStage, AstNodeImpl::asReduceTensorImpl(stageDef));
                        subFunctions.push_back(cfunc);
                        funcBody << genCUDAFuncCall(outStage, cfunc, streamArg, 1)<<";"<<std::endl;
                        pipelineStageName = cfunc.name;
                        break;
                    }
                    case NormNode: {
                        CFunc cfunc = generateNormCUDA(pipeline_, outStage, AstNodeImpl::asNormImpl(stageDef));
                        subFunctions.push_back(cfunc);
                        funcBody << genCUDAFuncCall(outStage, cfunc, streamArg, 1)<<";"<<std::endl;
                        pipelineStageName = cfunc.name;
                        break;
                    }
                    case MatMulNode: {
                        CFunc cfunc = generateCUBLASMatMul(pipeline_, outStage, AstNodeImpl::asMatMulImpl(stageDef));
                        subFunctions.push_back(cfunc);
                        funcBody << genCUDAFuncCall(outStage, cfunc, streamArg, 1)<<";"<<std::endl;
                        pipelineStageName = cfunc.name;
                        useCUBLAS = true;
                        break;
                    }
                    case DropoutNode: {
                        CFunc cfunc = generateDropoutCUDA(pipeline_, outStage, AstNodeImpl::asDropoutImpl(stageDef));
                        subFunctions.push_back(cfunc);
                        funcBody << genCUDAFuncCall(outStage, cfunc, streamArg, 1)<<";"<<std::endl;
                        pipelineStageName = cfunc.name;
                        break;
                    }
                    default:
                        ASSERT(false, "Case for type '" << AstNodeTypeToStr(stageDef->type()) << "' not defined.");
                }

                //Add as intermediate stage
                if (pipeline_.outputs().count(outStage) == 0 &&
                    pipeline_.explicitStoreLocations().count(outStage) == 0) {
                    if (stageDef->type() == ScatterNode) {
                        intermediateStages.push_back({outStage, false});
                    } else 
                        intermediateStages.push_back({outStage, true});
                }
            }
        } else {
            //Traverse the internal pipeline stage DAG to generate overlapped and/or fused stages
            if (pipeline_.name() == "model_parallel") {
                std::shared_ptr<StageImpl> matmulStage = nullptr;
                std::shared_ptr<StageImpl> rsStage = nullptr;
                std::shared_ptr<StageImpl> agStage = nullptr;
                pipelineStageName = "overlap";

                generateFusedNCCLCommColl(pipeline_, pipelineStage);

                for (auto outStage : pipelineStage->stages()) {
                    std::shared_ptr<ExpressionImpl> stageDef = outStage->definition();
                    if (stageDef->type() == MatMulNode) {
                        matmulStage = outStage;
                    } else if (stageDef->type() == ReduceScatterNode) {
                        rsStage = outStage;
                    } else if (stageDef->type() == AllGatherNode) {
                        agStage = outStage;
                    }
                }
                
                ASSERT(matmulStage != nullptr, "");
                CFunc cfunc = generateCUBLASMatMul(pipeline_, matmulStage, AstNodeImpl::asMatMulImpl(matmulStage->definition()));
                subFunctions.push_back(cfunc);
                useCUBLAS = true;
                intermediateStages.push_back({matmulStage, true});
                
                funcBody << genCUDAFuncCall(matmulStage, cfunc, streamArg, 1)<<";"<<std::endl;
                std::shared_ptr<ReduceScatterImpl> rsStageDef = AstNodeImpl::asReduceScatterImpl(rsStage->definition());
                std::shared_ptr<AllGatherImpl> agStageDef = AstNodeImpl::asAllGatherImpl(agStage->definition());
                funcBody << indent(1) << "ncclAllReduce(" << rsStageDef->arg()->name() << ", " << agStage->name() << ", " << 
                            genNumElem(agStageDef) << ", " << elemTypeToNCCLType(rsStageDef->arg()->elemType()) << "," << 
                            redOpToNCCLReduceOp(rsStageDef->reduceOp()) << ", " << commArg << ", " << streamArg << ");" << std::endl;
            } else {
                for (auto outStage : pipelineStage->stages()) {
                    std::shared_ptr<ExpressionImpl> stageDef = outStage->definition();
                    if (stageDef->isCommCollective()) {
                        hasACommCollStage = true;
                        break;
                    }
                }

                if (hasACommCollStage) {
                    funcBody << indent(1) << generateFusedNCCLCommColl(pipeline_, pipelineStage) << std::endl;
                    pipelineStageName = "FusedAllReduce";
                } else {
                    CFunc cfunc = generateFusedBinOpCodeCUDA(pipeline_, pipelineStage);
                    subFunctions.push_back(cfunc);
                    std::shared_ptr<StageImpl> sliceStage = nullptr;
                    for (auto stage : pipelineStage->stages()) {
                        if (stage->layout() == Sliced) {
                            sliceStage = stage;
                        }
                    }
                    if (sliceStage == nullptr) sliceStage = pipelineStage->stages()[0];
                    funcBody << genCUDAFuncCall(sliceStage, cfunc, streamArg, 1)<<";"<<std::endl;
                    for (auto liveout : pipelineStage->liveoutStages(pipeline_.outputs())) {
                        //Add liveouts that are not output as intermediate
                        if (pipeline_.outputs().count(liveout) == 0) {
                            intermediateStages.push_back({liveout, true});
                        }
                    }
                    for (auto intermediate : cfunc.intermediates) {
                        intermediateStages.push_back({intermediate, true});
                    }
                    pipelineStageName = cfunc.name;
                }
            }
        }

        psToNameAndTimeVar[pipelineStage] = std::make_pair(pipelineStageName, "elapsedTime" + pipelineStageName);
        funcBody << indent(1) << printEventRecord(stopEvent, "stream") << std::endl
                 << indent(1) << printEventSynchronize(stopEvent) << std::endl
                 << indent(1) << printEventElapsedTime(startEvent, stopEvent, elapsedTimeVar) << std::endl
                 << indent(1) << psToNameAndTimeVar[pipelineStage].second << " += " << elapsedTimeVar << ";" << std::endl
                 << std::endl;
    }

    if (options_ & GenMainFunction) {
        std::stringstream headers;

        headers << "#include \"header.h\"" << std::endl;
        os_ << headers.str();
    }

    std::set<std::shared_ptr<ExpressionImpl>> pipeArgs(pipeline_.outputs().begin(), pipeline_.outputs().end()); //set of inputs and outputs
    std::set<std::shared_ptr<ExpressionImpl>> outputArgs(pipeline_.outputs().begin(), pipeline_.outputs().end());
    //Remove all storeAt's target and add source

    for (auto it : pipeline_.explicitStoreLocations()) {
        if (pipeline_.outputs().count(it.first) > 0) {
            outputArgs.erase(it.first);
            outputArgs.insert(it.second);
        }
        
        if (!isStageAnIntermediate(intermediateStages, it.second)) {
            pipeArgs.erase(it.first);
            pipeArgs.insert(it.second);
        }
    }

    auto pipeDimExprs = allDimExprs(pipeArgs.begin(), pipeArgs.end());
    pipeArgs.insert(pipeDimExprs.begin(), pipeDimExprs.end());

    for (auto iter : pipeline_.arguments()) {
        pipeArgs.insert(iter);
    }
    
    /*Print sub functions (CUDA kernels)*/
    for (auto& cfunc : subFunctions) {
        os_ << cfunc.body << std::endl << std::endl;
    }
    
    /*Print Function name*/
    os_ << "void " << pipeline_.name() << "(";
    /*Function's arguments described in the pipeline, i.e., inputs and outputs*/
    for (auto arg : pipeArgs) {
        os_ << printArgument(arg) << ", ";
    }

    //Add intermediates as argument to pipeline function
    for (auto interm : intermediateStages) {
        os_ << printArgument(interm.stageImpl) << ", ";
    }
    //Time variables
    for (auto iter : psToNameAndTimeVar) {
        os_ << "float& " << iter.second.second << ", ";
    }
    //NCCLComm and CUDA Stream arguments
    os_ << ncclCommTy << " " << commArg << ", " << streamTy << " " << streamArg << ", " << commSizeTy << " " << commSizeArg << ", " << rankVarTy << " " << rankVar;

    if (useCUBLAS)
        os_ << ", " << cublasHandleTy << " " << cublasHandleVar;

    os_ << ")";
    /*Function body*/
    os_ << "{";

    os_ << std::endl;
    os_ << funcBody.str();
    os_ << std::endl;
    os_ << "}";
    os_ << std::endl;

    if (options_ & GenMainFunction) {
        std::stringstream mainFunc;
        const std::string mainSig = "int main(int argc, char** argv)";
        const std::string epochDecl = "if (argc < 2) { printf(\"Specify epochs as command arg\"); return 1;}\n"
                                      "   int epochs = atoi(argv[1]);\n";
        const std::string mpibarrier = "MPI_Barrier(MPI_COMM_WORLD);\n";
        const std::string ncclDestroyComm = "ncclCommDestroy(comm);\n";
        const std::string iterLoop = "for(int iter = 0; iter < epochs; iter++) {\n";
        std::stringstream mpiRefFunc;

        if (options_ & GenMultiProcessCode) {
            /* Generate reference implementation that uses MPI and do not 
             * follow fusion or storeAt.*/

            std::unordered_map<std::shared_ptr<ExpressionImpl>, std::shared_ptr<ExpressionImpl>> deviceExprToHostExpr;
            for (auto iter = pipeline_.arguments().begin(); 
                iter != pipeline_.arguments().end(); ++iter) {
                const std::shared_ptr<ExpressionImpl>& arg = *iter;
                deviceExprToHostExpr[arg] = std::shared_ptr<TensorImpl>(new TensorImpl (AstNodeImpl::asTensorImpl(arg)->copyWithNewName("__"+arg->name())));
            }
            
            std::stringstream mpiRefFuncCall;
            std::stringstream deviceToHostTransfers;

            if (options_ & GenResultsCheckCode) {
                mpiRefFunc << "bool mpiRef(";
                //Add declarations for input tensors and output from NCCL.
                for (auto iter = pipeline_.arguments().begin();
                    iter != pipeline_.arguments().end(); ++iter) {
                    const std::shared_ptr<ExpressionImpl>& arg = *iter;
                    TensorElemType customType = (arg->elemType() == Float16) ? Float32 : arg->elemType();
                    mpiRefFunc << printArgument(deviceExprToHostExpr[arg], customType) << ", ";
                    // if (arg->type() == StageNode or arg->type() == TensorNode)
                    //     mpiRefFunc << indent(1) << printCUDAMalloc(arg.get()) << std::endl;
                }

                for (auto stage : outputArgs) {
                    TensorElemType customType = (stage->elemType() == Float16) ? Float32 : stage->elemType();
                    mpiRefFunc << printArgument(stage, customType) << ", ";
                    // mainFunc << indent(1) << printCUDAMalloc(stage) << std::endl;
                }

                mpiRefFunc << "bool dummy=false)" << std::endl << "{" << std::endl;

                //Iterate over all stages in the Pipe and generate serial code.
                for (auto pipelineStage : pipeline_.topoOrder()) {
                    std::string stageName;
                    
                    for (auto outStage : pipelineStage->stages()) {
                        stageName = outStage->name();
                        
                        bool generateChecks = pipeline_.outputs().count(outStage) == 1;

                        //If there is only one stage
                        std::shared_ptr<ExpressionImpl> stageDef = outStage->definition();
                        bool hasMixedPrecision = false;
                        if (stageDef->type() == BinaryPointwiseOpNode) {
                            hasMixedPrecision = isMixedPrecision(AstNodeImpl::asBinaryPointwiseOp(stageDef)).size() > 0;
                        }
                        
                        if (hasMixedPrecision or stageDef->elemType() == Float16) {
                            //If using Mixed Precission of Float16 or the stage is of type Float16
                            //then generate Float32 code instead.
                            mpiRefFunc << indent(1) << printDeclaration(outStage, ";", "__", false, false, Float32) << std::endl;
                            mpiRefFunc << indent(1) << printNew(outStage, "__", Float32) << std::endl;
                        } else {
                            mpiRefFunc << indent(1) << printDeclaration(outStage, ";", "__", false, false) << std::endl;
                            mpiRefFunc << indent(1) << printNew(outStage, "__") << std::endl;
                        }
                        
                        switch (stageDef->type()) {
                            //Definition of collective communications contains only stage or tensor.
                            case AllReduceNode: {
                                std::shared_ptr<AllReduceImpl> allReduceColl = AstNodeImpl::asAllReduceImpl(stageDef);
                                mpiRefFunc << indent(1) << "MPI_Allreduce(" << "__" << allReduceColl->arg()->name() << ", " << "__" << stageName << ", " << 
                                    genNumElem(allReduceColl->arg()) << ", " << elemTypeToMPIType(allReduceColl->arg()->elemType()) << ", " << 
                                    redOpToMPIReduceOp(allReduceColl->reduceOp()) << ", MPI_COMM_WORLD);" << std::endl;
                                break;
                            }
                            case AllGatherNode: {
                                std::shared_ptr<AllGatherImpl> allGatherColl = AstNodeImpl::asAllGatherImpl(stageDef);
                                mpiRefFunc << indent(1) << "MPI_AllGather(" << allGatherColl->arg()->name() << ", " << stageName << ", " << 
                                    genNumElem(allGatherColl->arg()) << ", " << elemTypeToMPIType(allGatherColl->arg()->elemType()) << 
                                    ", MPI_COMM_WORLD);" << std::endl;
                                break;
                            }
                            case ReduceScatterNode: {
                                std::shared_ptr<ReduceScatterImpl> reduceScatterColl = AstNodeImpl::asReduceScatterImpl(stageDef);
                                mpiRefFunc << indent(1) << "MPI_Reduce_Scatter(" << reduceScatterColl->arg()->name() << ", " << stageName << ", " << 
                                    genNumElem(reduceScatterColl->arg()) << ", " << elemTypeToMPIType(reduceScatterColl->arg()->elemType()) << 
                                    ", " << redOpToMPIReduceOp(reduceScatterColl->reduceOp()) << ", MPI_COMM_WORLD);" << std::endl;
                                break;
                            }
                            case ReduceNode: {
                                // std::shared_ptr<ReduceImpl> reduceColl = AstNodeImpl::asReduceImpl(stageDef);
                                // funcBody << indent(1) << "ncclReduce(" << reduceColl->arg()->name() << ", " << stageName << ", " << 
                                //     reduceColl->arg()->nelem() << ", " << elemTypeToNCCLType(reduceColl->arg()->elemType()) << 
                                //     redOpToNCCLReduceOp(reduceColl->reduceOp()) << ", " << reduceColl->root() << ", " <<
                                //     commArg << ", " << streamArg << ");" << std::endl;
                                ASSERT(false, "To implement");
                                break;
                            }
                            case BroadcastNode: {
                                // std::shared_ptr<BroadcastImpl> broadcastColl = AstNodeImpl::asBroadcastImpl(stageDef);
                                // funcBody << indent(1) << "ncclBroadcast(" << broadcastColl->arg()->name() << ", " << stageName << ", " << 
                                //     broadcastColl->arg()->nelem() << ", " << elemTypeToNCCLType(broadcastColl->arg()->elemType()) << 
                                //     ", " << broadcastColl->root() << ", " <<
                                //     commArg << ", " << streamArg << ");" << std::endl;
                                ASSERT(false, "To implement");
                                break;
                            }
                            case BinaryPointwiseOpNode:
                            case UnaryPointwiseOpNode:
                            case IteNode: {
                                std::string c = generateOpCodeCPU(pipeline_, outStage, 
                                                                    stageDef,
                                                                    generateChecks);
                                mpiRefFunc << c;
                                break;
                            }
                            case ReduceTensorNode: {
                                std::string c = generateReduceTensorCodeCPU(pipeline_, outStage, 
                                                                            AstNodeImpl::asReduceTensorImpl(stageDef).get(),
                                                                            generateChecks);
                                mpiRefFunc << c;
                                break;
                            }
                            case ScatterNode: {
                                //A Scatter Node 
                                // std::shared_ptr<ScatterImpl> scatter = AstNodeImpl::asScatterImpl(stageDef);
                                // funcBody << indent(1) << stageName << " = " << scatter->arg()->name() << " + " << 
                                //     scatter->size(0) << "/" << scatter->numGPUs() << " * " << rankString << std::endl;
                                ASSERT(false, "To implement");
                                break;
                            }
                            default:
                                ASSERT(false, "Case for type '" << stageDef->type() << "' not defined.");
                        }
                    }
                }

                mpiRefFunc << indent(1) << "return true;" << std::endl << "}" << std::endl;
                //Copy from device to host for outputs only
                for (auto iter = pipeline_.arguments().begin(); 
                    iter != pipeline_.arguments().end(); ++iter) {
                    const std::shared_ptr<ExpressionImpl>& arg = *iter;
                    if (arg.get()->elemType() == Float16) {
                        deviceToHostTransfers << indent(2) << printDeclaration(deviceExprToHostExpr[arg], ";", "", false, false, Float32) << ";" << std::endl;
                        deviceToHostTransfers << indent(2) << "if (iter == 0) {"<<std::endl;
                        if (arg.get()->type() != VariableNode) {
                            deviceToHostTransfers << indent(2) << printNew(deviceExprToHostExpr[arg], "", Float32) << ";" << std::endl;
                            deviceToHostTransfers << indent(2) << printCUDAMemcpyHalfD2FloatH(deviceExprToHostExpr[arg], arg) << ";" << std::endl;
                        }
                        else {
                            deviceToHostTransfers << indent(2) << deviceExprToHostExpr[arg]->name() << " = __half2float(" << arg.get()->name() << ");" << std::endl;
                        }
                        deviceToHostTransfers << indent(2) << "}"<<std::endl;
                    } else {
                        deviceToHostTransfers << indent(2) << printDeclaration(deviceExprToHostExpr[arg]) << ";" << std::endl;
                        deviceToHostTransfers << indent(2) << "if (iter == 0) {"<<std::endl;

                        if (arg.get()->type() != VariableNode) {
                            deviceToHostTransfers << indent(2) << printNew(deviceExprToHostExpr[arg]) << ";" << std::endl;
                            deviceToHostTransfers << indent(2) << printCUDAMemcpyD2H(deviceExprToHostExpr[arg], arg) << ";" << std::endl;
                        }
                        else
                            deviceToHostTransfers << indent(2) << deviceExprToHostExpr[arg]->name() << " = " << arg.get()->name() << ";" << std::endl;
                        deviceToHostTransfers << indent(2) << "}"<<std::endl;
                    }
                    // if (arg->type() == StageNode or arg->type() == TensorNode)
                    //     mpiRefFunc << indent(1) << printCUDAMalloc(arg.get()) << std::endl;
                }
                mpiRefFuncCall << indent(2) << "if (iter == 0) assert(mpiRef(";
                for (auto iter = pipeline_.arguments().begin(); 
                    iter != pipeline_.arguments().end(); ++iter) {
                    const std::shared_ptr<ExpressionImpl>& arg = *iter;
                    mpiRefFuncCall << deviceExprToHostExpr[arg]->name() << ", ";
                }

                for (auto stage : outputArgs) {
                    mpiRefFuncCall << stage->name() << ", ";
                }

                mpiRefFuncCall << "false));" << std::endl;
            }

            mainFunc << mainSig;
            //Start main function block
            mainFunc << "{" << std::endl;
            
            const std::string mpiStartCode = 
                "  //Get number of gpus in the node\n"
                "  int N_GPUs;\n"
                "  CUDACHECK(cudaGetDeviceCount(&N_GPUs));\n"
                "  MPI_Init(&argc, &argv);\n"
                "  int comm_size, rank;\n"
                "  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);\n"
                "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n"
                "  ncclComm_t comm;\n"
                "  CUDACHECK(cudaSetDevice(rank % N_GPUs));\n"
                "  //initializing NCCL\n"
                "  ncclUniqueId id;\n"
                "  if (rank == 0) ncclGetUniqueId(&id);\n"
                "  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);\n"
                "  ncclCommInitRank(&comm, comm_size, id, rank);\n";
            const std::string streamDecl = "  " + streamTy + " " + streamArg + ";\n" + "  cudaStreamCreate(&"+streamArg+");\n";
            const std::string cublasHandleDecl = indent(1) + cublasHandleTy + " "  + cublasHandleVar + ";\n" +
                                                 indent(1) + cublasCheck("cublasCreate(&" + cublasHandleVar+")") + "\n" +
                                                 indent(1) + cublasCheck("cublasSetStream(" + cublasHandleVar + ", " + streamArg+")") + "\n" +
                                                 indent(1) + cublasCheck("cublasSetMathMode(" + cublasHandleVar + ", CUBLAS_TENSOR_OP_MATH)");

            mainFunc << mpiStartCode << "  " << epochDecl << streamDecl;
            if (useCUBLAS)
                mainFunc << cublasHandleDecl << std::endl;
            mainFunc << indent(1) << mpibarrier << std::endl;
            int indentLevel = 1;
            
            //For loop for different values of sizes to evaluate
            if (varBounds.size() == 0) {
                mainFunc << indent(indentLevel) << "for (int __i = 10; __i < 30; __i++) {" << std::endl;
                indentLevel++;
                mainFunc << indent(indentLevel) << "size_t N = 1 << __i;" << std::endl;
            } else {
                for (auto varBound : varBounds) {
                    if (varBound.type_ == Values) {
                        if (varBound.values_.size() == 1)
                            mainFunc << indent(indentLevel) << "size_t " << varBound.var_.impl()->name() << " = " << varBound.values_[0] << ";" << std::endl;
                        else {
                            std::string varName = varBound.var_.impl()->name();
                            std::string arrayVar = "array_"+varName;
                            mainFunc << indent(indentLevel) << "int " << arrayVar << "[] = {";
                            for (int i = 0; i < varBound.values_.size(); i++) {
                                mainFunc << varBound.values_[i];
                                if (i != varBound.values_.size() - 1)
                                    mainFunc << ", ";
                            }
                            mainFunc << "};" << std::endl;
                            std::string iteratorVar = "iter_"+varName;
                            mainFunc << indent(indentLevel) << "for (int " << iteratorVar << " = 0" << "; " << 
                                        iteratorVar << "< " << "sizeof(" << arrayVar << ")/sizeof(" << arrayVar<<"[0]" << ");" << 
                                        iteratorVar << "++) {" << std::endl;
                            indentLevel++;
                            mainFunc << indent(indentLevel) << "int " << varName << " = " << arrayVar << "[" << iteratorVar << "];" << std::endl;
                        }
                    }
                }
            }
            mainFunc << indent(indentLevel) << "// Inputs" << std::endl;
            //Add declarations for tensors and allocate memory.
            for (auto iter = pipeline_.arguments().begin(); 
                iter != pipeline_.arguments().end(); ++iter) {
                const std::shared_ptr<ExpressionImpl>& arg = *iter;
                mainFunc << indent(indentLevel) << printDeclaration(arg) << std::endl;
                if (arg->type() == StageNode or arg->type() == TensorNode) {
                    mainFunc << indent(indentLevel) << printCUDAMalloc(arg, hasACommCollStage) << std::endl;
                    mainFunc << indent(indentLevel) << "cudaMemRandInt(" << arg->name() << ", " << genNumElem(arg) << ");" << std::endl;
                } else {
                    mainFunc << indent(indentLevel) << arg->name() << " = 1.0f;" << std::endl;
                }
            }

            mainFunc << std::endl << indent(indentLevel) << "// Outputs" << std::endl;
            //Add declarations for output tensors and allocate memory.
            for (auto stage : outputArgs) {
                if (!pipeline_.isArgument(stage.get())) {
                    mainFunc << indent(indentLevel) << printDeclaration(stage) << std::endl;
                    mainFunc << indent(indentLevel) << printCUDAMalloc(stage) << std::endl;
                }
            }

            std::string intermDeclCode, intermAllocCode, intermFreeCode;
            printIntermdiatesCUDAAlloc(intermediateStages, intermDeclCode, intermAllocCode, 
                                       intermFreeCode, indentLevel);
            if (intermediateStages.size() > 0) {
                mainFunc << std::endl << indent(indentLevel) << "// Intermediates" << std::endl;
                mainFunc << intermDeclCode;
                mainFunc << intermAllocCode;
            }

            //Declare time variables
            for (auto iter : psToNameAndTimeVar) {
                mainFunc << indent(indentLevel) << "float " << iter.second.second << " = 0;" << std::endl;
            }

            //Start epochs loop
            mainFunc << indent(indentLevel) << iterLoop;
            mainFunc << deviceToHostTransfers.str();
            //Add timing information
            //Declare events
            indentLevel++;
                            
            //Print pipe function call
            mainFunc << indent(indentLevel) << pipeline_.name() << "(";
            for (auto arg : pipeArgs) {
                mainFunc << arg->name() << ", ";
            }
            for (auto interm : intermediateStages) {
                mainFunc << interm.stageImpl->name() << ", ";
            }
            for (auto iter : psToNameAndTimeVar) {
                mainFunc << iter.second.second << ", ";
            }
            mainFunc << commArg << ", " << streamArg << ", " << commSizeArg << ", " << rankVar;
            if (useCUBLAS)
                mainFunc << ", " << cublasHandleVar;
            mainFunc << "); "<< std::endl;
            
            //Add mpiref function call
            mainFunc << mpiRefFuncCall.str();
            //End epochs loop
            indentLevel--;
            mainFunc << indent(indentLevel) << "}" << std::endl;
            for (auto arg : pipeline_.arguments()) {
                if (arg->type() == StageNode or arg->type() == TensorNode) {
                    mainFunc << indent(indentLevel) << printCUDAFree(arg) << std::endl;
                }
            }
            for (auto stage : outputArgs) {
                if (!pipeline_.isArgument(stage.get())) {
                    mainFunc << indent(indentLevel) << printCUDAFree(stage) << std::endl;
                }
            }
            mainFunc << intermFreeCode;
            std::stringstream printfTimeString;
            if (varBounds.size() == 0)
                printfTimeString << "if (rank == 0) " << std::endl << indent(indentLevel+1) << "printf(\"{SZ: %ld, Epochs: %d, ";
            else {
                std::string varBoundsPrintfFmt = "";
                for (auto varBound : varBounds) {
                    varBoundsPrintfFmt += varBound.var_.impl()->name() + ": %ld, ";
                }
                printfTimeString << "if (rank == 0) " << std::endl << indent(indentLevel+1) << "printf(\"{" << varBoundsPrintfFmt << "Epochs: %d, ";
            }

            for (auto iter : psToNameAndTimeVar) {
                printfTimeString << iter.second.first << ": %f, ";
            }
            
            printfTimeString << "Total: %f}\\n\", ";

            if (varBounds.size() == 0)
                printfTimeString << "N, " << "epochs, ";
            else {
                std::string vars = "";
                for (auto varBound : varBounds)
                    vars += varBound.var_.impl()->name() + ", ";
                printfTimeString << vars << "epochs, ";
            }

            for (auto iter : psToNameAndTimeVar) {
                printfTimeString << iter.second.second << ", ";
            }
            for (auto iter = psToNameAndTimeVar.begin(); iter != psToNameAndTimeVar.end();) {
                printfTimeString << iter->second.second;
                if (++iter != psToNameAndTimeVar.end())
                    printfTimeString << " + ";
            }
            printfTimeString << ");" << std::endl;
            mainFunc << indent(indentLevel) << printfTimeString.str();
            //End variables for loop
             if (varBounds.size() == 0) {
                indentLevel--;
                mainFunc << indent(indentLevel) << "}" << std::endl;
            } else {
                for (auto varBound : varBounds) {
                    if (varBound.type_ == Values) {
                        if (varBound.values_.size() > 1) {
                            indentLevel--;
                            mainFunc << indent(indentLevel) << "}" << std::endl;
                        }
                    }
                }
            }
            //End main function block
            mainFunc << indent(indentLevel) << "MPI_Finalize();" << std::endl
                     << "}" << std::endl;
        } else if (options_ & GenSingleProcessCode) {

        } else {
            ASSERT(false, "None of GenMultiProcessCode or GenSingleProcessCode mentioned");
        }

        os_ << mpiRefFunc.str();
        os_ << mainFunc.str();
    }
}

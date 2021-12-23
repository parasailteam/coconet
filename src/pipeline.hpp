#ifndef __PIPELINE_HPP__
#define __PIPELINE_HPP__

#include "dsl.hpp"
#include "ast.hpp"
#include "astvisitor.hpp"

#include <queue>
#include <stack>
#include <unordered_map>
#include <set>
#include <vector>
#include <unordered_set>
#include <fstream>

template<class T, class U>
std::vector<typename T::value_type> setIntersection(const T& u, const U& v) 
{
    std::vector<typename T::value_type> intersection;
    ASSERT(std::is_sorted(u.begin(), u.end()), "Elements are required to be in sorted order");
    ASSERT(std::is_sorted(v.begin(), v.end()), "Elements are required to be in sorted order");
    std::set_intersection(u.begin(), u.end(), v.begin(), v.end(), 
                          std::back_inserter(intersection));
    return intersection;
}

template<class T, class U>
std::vector<typename T::value_type> setDifference(const T& u, const U& v) 
{
    std::vector<typename T::value_type> difference;
    ASSERT(std::is_sorted(u.begin(), u.end()), "Elements are required to be in sorted order");
    ASSERT(std::is_sorted(v.begin(), v.end()), "Elements are required to be in sorted order");
    std::set_difference(u.begin(), u.end(), v.begin(), v.end(), 
                          std::back_inserter(difference));
    return difference;
}


template<class T, class U>
bool hasIntersection(const T& u, const U& v)
{
    return !setIntersection(u, v).empty();
}

namespace ACCCDSL {
    using namespace ACCCDSLImpl;
    class Pipeline;
    class PipelineStage;
    class Autotuner;

    enum SplitType {
        AllReduceRSAG,
        AllReduceBR,
    };

    class Pipeline;
    class PipelineTests;
    enum StageStoreLocation {
        Register,
        Memory,
    };

    enum CodeGenOptions {
        GenResultsCheckCode = 1, //Generate code to check obtained results
        GenMainFunction = 1 << 1, //Generate main function for execution
        GenMultiProcessCode = 1 << 2, //Generate a multiprocess code that uses MPI
        GenSingleProcessCode = 1 << 3, //Generate single process code that uses nccl group
    };

    enum PipelineStageType {
        Fused,
        Overlapped,
        Single
    };

    struct ReorderedStages {
        std::vector<Stage> compStages;
        std::vector<Stage> allGatherStages;
    };

    class PipelineStage {
    private:
        std::set<PipelineStage*> parents_; //Parent's liveouts are the inputs for current stage
        std::set<PipelineStage*> children_; //Current Stage's liveouts are the inputs for child stages
        std::vector<std::shared_ptr<StageImpl>> stages_; //All stages in the fused stage ensured to be in topological order
        struct FuseOverlapNode {
            //Represents internal DAG that have a combination of overlap and fused stages
            PipelineStageType type;
            std::vector<std::shared_ptr<StageImpl>> stages;
            std::shared_ptr<FuseOverlapNode> parent;
            std::shared_ptr<FuseOverlapNode> child;
        };
        
        std::shared_ptr<FuseOverlapNode> fuseOverlapDAG;
        int topoOrder_;
        std::unordered_map<std::shared_ptr<StageImpl>, StageStoreLocation> stageStoreLoc_;
        CollCommOperationType fusedIntoCollComm_;
        PipelineStageType type_;
        friend Pipeline;

    public:
        PipelineStage(std::shared_ptr<StageImpl> stage) : PipelineStage(std::vector<std::shared_ptr<StageImpl>>{stage}){}
        PipelineStage(std::vector<std::shared_ptr<StageImpl>> stages) : stages_(stages), fusedIntoCollComm_(NoneCollCommOp), type_(Single), fuseOverlapDAG(nullptr) {} 

        PipelineStageType type() {return type_;}
        void setType(PipelineStageType _type) {type_ = _type;}
        std::set<ExpressionImpl*> usedExprs() 
        {
            InputsVisitor inputsVisitor;
            std::set<ExpressionImpl*> used;
            
            for (auto stage : stages_) {
                auto _ = inputsVisitor.inputs(*stage);
                used.insert(_.begin(), _.end());
            }
            return used;
        }
        void removeStage(std::shared_ptr<StageImpl> stage) 
        {
            auto iter = std::find(stages_.begin(), stages_.end(), stage);
            if (iter != stages_.end())
                stages_.erase(iter);
            FuseOverlapNode* node = fuseOverlapDAG.get();

            while (node != nullptr) {
                auto iter = std::find(node->stages.begin(), node->stages.end(), stage);
                if (iter != stages_.end()) 
                    node->stages.erase(iter);
                node = node->child.get();
                // os << "}" << std::endl;
            }
        }
        void fuseOrOverlapStages(PipelineStage* prevPS, PipelineStageType combinationType)
        {
            std::shared_ptr<FuseOverlapNode> newHead;
            std::vector<std::shared_ptr<StageImpl>> _stages;
            if (prevPS->fuseOverlapDAG == nullptr) {
                //Handle first fusion/overlap of 2 or more nodes by creating a new fuse/overlap DAG in prevPS
                _stages = std::vector<std::shared_ptr<StageImpl>>(stages());
                auto prevStages = prevPS->stages();
                //Adding at end will maintain the topological order
                _stages.insert(_stages.end(), prevStages.begin(), prevStages.end());
            } else {
                _stages = stages();
            }

            newHead = std::shared_ptr<FuseOverlapNode>(new FuseOverlapNode{combinationType, _stages, nullptr, prevPS->fuseOverlapDAG});
            if (prevPS->fuseOverlapDAG)
                prevPS->fuseOverlapDAG->parent = newHead;
            fuseOverlapDAG = newHead;
        }

        bool usesExpr(ExpressionImpl* expr) 
        {
            auto _usedExprs = usedExprs();

            return _usedExprs.count(expr) > 0;
        }

        std::set<std::shared_ptr<ExpressionImpl>> usedExprsAsSharedPtr()
        {
            std::set<std::shared_ptr<ExpressionImpl>> used;
            std::queue<std::shared_ptr<AstNodeImpl>> exprQueue;
            std::unordered_set<std::shared_ptr<AstNodeImpl>> visitedExprs;

            for (auto stage : stages_) {
                for (auto child : stage->children())
                    exprQueue.push(child);
            }

            while (!exprQueue.empty()) {
                auto expr = exprQueue.front();
                exprQueue.pop();
                
                if (visitedExprs.count(expr) > 0)
                    continue;

                if (expr->type() == StageNode || expr->type() == TensorNode) {
                    used.insert(AstNodeImpl::asExpressionImpl(expr));
                } else {
                    for (auto child : expr->children())
                        exprQueue.push(child);
                }

                visitedExprs.insert(expr);
            }

            return used;
        }

        std::set<std::shared_ptr<StageImpl>> usedStages()
        {
            std::set<std::shared_ptr<StageImpl>> used;
           
            for (auto expr : usedExprsAsSharedPtr()) {
                if (expr->type() == StageNode) {
                    used.insert(AstNodeImpl::asStageImpl(expr));
                }
            }

            return used;
        }

        void _replaceExpr(std::shared_ptr<AstNodeImpl> parent, 
                          std::shared_ptr<AstNodeImpl> expr, std::shared_ptr<AstNodeImpl> replacement) 
        {
            if (parent->hasChild(expr)) {
                if (parent->type() == UpdateNode) {
                    //Cannot replace arg of UpdateImpl
                    if (AstNodeImpl::asUpdateImpl(parent)->arg() != expr) {
                        auto iter = parent->findChild(expr);
                        parent->replaceChildren(iter, replacement);
                    }
                } else {
                    auto iter = parent->findChild(expr);
                    parent->replaceChildren(iter, replacement);
                }
            }

            for (auto child : parent->children()) {
                if (child != replacement) {
                    _replaceExpr(child, expr, replacement);
                }
            }
        }
        void replaceExpr(std::shared_ptr<AstNodeImpl> expr, std::shared_ptr<AstNodeImpl> replacement)
        {
            for (auto stage : stages_) {
                _replaceExpr(stage, expr, replacement);
            }
        }
        void _setupAndCheckDimensions(std::shared_ptr<AstNodeImpl> node)
        {
            for (auto child : node->children()) {
                _setupAndCheckDimensions(child);
            }
            if (dynamic_cast<ExpressionImpl*>(node.get()) != nullptr) {
                AstNodeImpl::asExpressionImpl(node)->setupAndCheckDimensions();
            }
        }
        void setupAndCheckDimensions()
        {
            for (auto stage : stages_)
                _setupAndCheckDimensions(stage);
        }

        std::set<ExpressionImpl*> inputExprs()
        {
            InputsVisitor inputsVisitor;
            std::set<ExpressionImpl*> inputExprs;

            //Add all inputs to each stage
            for (auto stage : stages_) {
                //For now we only support binary pointwise ops
                switch (stage->definition()->type()) {
                    case BinaryPointwiseOpNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<BinaryPointwiseOp*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }
                    case UnaryPointwiseOpNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<UnaryPointwiseOp*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }
                    case IteNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<IteImpl*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }

                    case AllReduceNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<AllReduceImpl*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }

                    case ReduceScatterNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<ReduceScatterImpl*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }
                    case AllGatherNode: {
                        std::set<ExpressionImpl*> binOpInputs = inputsVisitor.inputs(*dynamic_cast<AllGatherImpl*>(stage->definition().get()));
                        inputExprs.insert(binOpInputs.begin(), binOpInputs.end());
                        break;
                    }
                    default:
                        std::cout << "TOOD: Need to handle " << AstNodeTypeToStr(stage->definition()->type()) << std::endl;
                }
            }

            //Remove fused stages
            for (auto stage : stages_) {
                inputExprs.erase(stage.get());
            }

            return inputExprs;
        }

        void fusedIntoCollComm(CollCommOperationType collComm) {fusedIntoCollComm_ = collComm;}
        CollCommOperationType getFusedIntoCollComm() {return fusedIntoCollComm_;}
        StageStoreLocation getStorageLocation(std::shared_ptr<StageImpl> stage) 
        {
            return stageStoreLoc_.at(stage);
        }

        std::set<std::shared_ptr<StageImpl>> liveoutStages(const std::set<std::shared_ptr<StageImpl>>& pipelineOutputs)
        {
            std::set<std::shared_ptr<StageImpl>> liveouts;
            std::set<std::shared_ptr<StageImpl>> setStages(stages().begin(), stages().end());

            for (auto child : children()) {
                //If a stage of pipeStage is used by any of its child then
                //it is a liveout.
                for(auto stage : setIntersection(child->usedExprsAsSharedPtr(), setStages)) {
                    ASSERT(stage->type() == StageNode, "Should be stage.");
                    liveouts.insert(AstNodeImpl::asStageImpl(stage));
                }
            }

            std::vector<std::shared_ptr<StageImpl>> intersection = setIntersection(pipelineOutputs, setStages);
            liveouts.insert(intersection.begin(), intersection.end());
            
            return liveouts;
        }

        std::set<std::shared_ptr<StageImpl>> liveinExprs()
        {
            //Get all the live in expressions to the stage.
            std::set<std::shared_ptr<StageImpl>> liveins;
            std::set<std::shared_ptr<StageImpl>> setStages(stages().begin(), stages().end());

            for (auto stage : stages_) {
                auto usedExprs = stage->childrenOfType<StageImpl>();
                // std::cout << std::endl;
                // std::cout << std::endl;
                // for (auto e : usedExprs) std::cout << this << " u " << e.get() << std::endl;
                // for (auto e : setStages) std::cout << this << " s " << e.get() << std::endl;
                //An expression is livein if it is not generated in any of the other stages
                auto difference = setDifference(usedExprs, setStages);
                // for (auto e : difference) std::cout << this << " d " << e.get() << std::endl;
                // std::cout << std::endl;
                // std::cout << std::endl;
                liveins.insert(difference.begin(), difference.end());
            }

            return liveins;
        }

        void setStorageLocation(const std::set<std::shared_ptr<StageImpl>>& pipelineOutputs)
        {     
            auto liveouts = liveoutStages(pipelineOutputs);
            auto liveins = liveinExprs();

            for (auto stage : liveouts) {
                //All liveouts are stored in memory
                stageStoreLoc_[stage] = Memory;
            }

            for (auto stage : liveins) {
                //All inputs are stored in memory
                stageStoreLoc_[stage] = Memory;
            }

            for (auto stage : stages()) {
                if (liveouts.count(stage) == 0)
                    stageStoreLoc_[stage] = Register;
            }

            //If any stage has an update node then set its location to memory
            for (auto stage : stages()) {
                if (stage->definition()->type() == UpdateNode)
                    stageStoreLoc_[stage] = Memory;
            }
        }

        void setTopoOrder(int topoOrder) {topoOrder_ = topoOrder;}
        int getTopoOrder() const {return topoOrder_;}

        void addChild(PipelineStage* child) {children_.insert(child);}
        void addChildren(std::vector<PipelineStage*>::iterator begin, std::vector<PipelineStage*>::iterator end) 
        {children_.insert(begin, end);}
        void addChildren(const std::set<PipelineStage*>& children) 
        {children_.insert(children.begin(), children.end());}
        void addParent(PipelineStage* parent) {parents_.insert(parent);}
        void addParents(const std::set<PipelineStage*>& parents) 
        {parents_.insert(parents.begin(), parents.end());}
        bool hasChild(PipelineStage* child) 
        {return children_.find(child) != children_.end();}
        bool hasParent(PipelineStage* parent) 
        {return parents_.find(parent) != parents_.end();}

        void replaceChild(PipelineStage* child, PipelineStage* replacement) 
        {
            auto iter = children_.find(child);
            children_.erase(iter);
            children_.insert(replacement);
        }
        void replaceParent(PipelineStage* parent, PipelineStage* replacement) 
        {
            auto iter = parents_.find(parent);
            parents_.erase(iter);
            parents_.insert(replacement);
        }

        void removeChild(PipelineStage* child) {children_.erase(child);}
        void removeParent(PipelineStage* parent) {parents_.erase(parent);}

        size_t numChildren() {return children_.size();}
        size_t numParent() {return parents_.size();}

        bool isPipelineInput() {return numParent() == 0;}
        bool isPipelineOutput(const std::set<std::shared_ptr<StageImpl>>& stageOutputs) 
        {
            for (auto stage : stages()) {
                if (stageOutputs.count(stage) > 0)
                    return true;
            }

            return false;
        }

        void addStage(std::shared_ptr<StageImpl> stage) {stages_.push_back(stage);}
        const std::vector<std::shared_ptr<StageImpl>>& stages() {return stages_;}

        const std::set<PipelineStage*>& children () {return children_;}
        const std::set<PipelineStage*>& parents () {return parents_;}

        void print(std::ostream& os) {
            SingleStagePrintVisitor printer(os);
            if (type() == Single) {
                for (auto s : stages()) {
                    printer.print(*s);
                }
            } else {
                FuseOverlapNode* node = fuseOverlapDAG.get();
                // std::stack<FuseOverlapNode*> nodeStack;
                //FIXME: do this recursively in a function
                while (node != nullptr) {
                    if (node->type == Fused) {
                        os << "fused {" << std::endl;
                    } else if (type() == Overlapped) {
                        os << "overlapped {" << std::endl;
                    }
                    
                    for (auto s : node->stages) {
                        printer.print(*s);
                    }

                    node = node->child.get();
                    os << "}" << std::endl;
                }

                // if (type() == Fused) {
                //     os << "}" << std::endl;
                // } else if (type() == Overlapped) {
                //     os << "}" << std::endl;
                // }
            }
        }
    };

    enum CodeGenVarBoundsType {
        IncrementRange,
        LogRange,
        Values
    };

    struct CodeGenVarBounds {        
        Variable var_;
        CodeGenVarBoundsType type_;
        int start_;
        int end_;
        int increment_;
        
        std::vector<int> values_;
        
        CodeGenVarBounds(Variable var, int start, int end, int increment, CodeGenVarBoundsType type) : var_(var),  start_(start), end_(end), increment_(increment), type_(type) {}
        CodeGenVarBounds(Variable var, std::vector<int> values) : var_(var), values_(values), type_(Values) {}
    };

    class Pipeline {
        protected:
        std::set<std::shared_ptr<StageImpl>> stageOutputs_;
        std::unordered_map<std::shared_ptr<StageImpl>, PipelineStage*> dslStageToPipelineStage;
        std::unordered_map<AstNodeImpl*, std::shared_ptr<AstNodeImpl>> astPtrToSharedPtr;
        std::vector<PipelineStage*> inputs_;
        std::vector<PipelineStage*> outputs_;
        std::vector<std::shared_ptr<ExpressionImpl>> arguments_;
        std::vector<PipelineStage*> topoOrder_;
        std::string name_;
        std::unordered_map<std::shared_ptr<StageImpl>, std::shared_ptr<ExpressionImpl>> explicitStoreLocations_;
        bool _storeAt(StageImpl* s, ExpressionImpl* t);
        std::unordered_map<AstNodeImpl*, std::shared_ptr<AstNodeImpl>> origToCloneMap_;
        friend Autotuner;

        public:
        friend PipelineTests;
        Pipeline(std::vector<Expression> args, std::vector<Stage> output) : Pipeline("pipe", args, output) {}
        Pipeline(std::string name, std::vector<Expression> args, std::vector<Stage> output) : name_(name)
        {
            for (auto arg : args) {
                arguments_.push_back(arg.impl());
                ASSERT(arg.impl()->type() == TensorNode || arg.impl()->type() == VariableNode, "Invalid argument " << arg.impl()->name() << "Argument can only be Tensor or Variable");
            }
            for (auto stage : output) {
                stageOutputs_.insert(AstNodeImpl::asStageImpl(stage.impl()));
            }

            createDAG();
            updateExplicitStorageLocations();
        }

        std::vector<PipelineStage*> inputs() {return inputs_;}
        std::string name() {
            auto n = replaceAllSubString(name_, "-", "_");
            return replaceAllSubString(n, " ", "_");
        }
        const std::vector<PipelineStage*>& topoOrder() {return topoOrder_;}
        const std::set<std::shared_ptr<StageImpl>>& outputs() {return stageOutputs_;}
        const std::vector<std::shared_ptr<ExpressionImpl>>& arguments() {return arguments_;} 
        bool isArgument(ExpressionImpl* arg) {
            for (auto _arg : arguments_) {
                if (arg == _arg.get())
                    return true;
            }
            return false;
        }
        const std::unordered_map<std::shared_ptr<StageImpl>, std::shared_ptr<ExpressionImpl>>& explicitStoreLocations() {return explicitStoreLocations_;}

        void createDAG();
        
        /**Transformations**/
        void fuse(std::vector<Stage> stagesToFuse);
        void overlap(std::vector<Stage> stagesToOverlap);
        void asSlice(std::vector<Tensor> replicatedInputs);
        void asSlice(Tensor replicatedInput) {asSlice(std::vector<Tensor>({replicatedInput}));}
        //ReduceScatter stage and allgather stage
        std::pair<Stage, Stage> split(Stage stageToSplit, SplitType splitType);
        //New comp stages and new allgather stages
        ReorderedStages reorder(std::vector<Stage> comps, Stage allGather);
        void fuseInto(std::vector<Expression> stages, CollCommOperationType t);
        /******************/
        void fuse(std::vector<std::shared_ptr<StageImpl>> stagesToFuse);
        void overlap(std::vector<std::shared_ptr<StageImpl>> stagesToOverlap);
        std::pair<std::shared_ptr<StageImpl>, std::shared_ptr<StageImpl>> split(std::shared_ptr<StageImpl> stageToSplit, SplitType splitType);
        std::pair<std::vector<std::shared_ptr<StageImpl>>, std::vector<std::shared_ptr<StageImpl>>> reorder(std::vector<std::shared_ptr<StageImpl>> comps, std::shared_ptr<StageImpl> allGather);
        void asSlice(std::vector<std::shared_ptr<TensorImpl>> replicatedInput);
        PipelineStage* combineStagesInDAG(std::vector<std::shared_ptr<StageImpl>> stagesToCombine, PipelineStageType combinationType);
        bool checkIfAllStagesAreFused(std::vector<Stage> comps);
        void createTopologicalSort();
        void updateExplicitStorageLocations();

        template<typename T>
        std::shared_ptr<T> sharedPtrForAstPtr(T* astPtr)
        {
            return std::dynamic_pointer_cast<T>(astPtrToSharedPtr.at(astPtr));
        }

        template<class InputIterator>
        void addPipelineStageChildren(PipelineStage* pipelineStage, InputIterator first, InputIterator last) {
            for (auto it = first; it < last; it++) {
                PipelineStage* ps = createOrGetPipelineStageForDslStage(*it);
                pipelineStage->addChild(ps);
            }
        }

        template<class InputIterator>
        void addPipelineStageParents(PipelineStage* pipelineStage, InputIterator first, InputIterator last) {
            for (auto it = first; it != last; it++) {
                PipelineStage* ps = createOrGetPipelineStageForDslStage(*it);
                pipelineStage->addParent(ps);
            }
        }

        PipelineStage* createOrGetPipelineStageForDslStage(std::shared_ptr<StageImpl> stage) {
            if (dslStageToPipelineStage.find(stage) == dslStageToPipelineStage.end()) {
                PipelineStage* ps = new PipelineStage(stage);
                dslStageToPipelineStage[stage] = ps;
                return ps;
            }

            return dslStageToPipelineStage[stage];
        }

        std::set<std::shared_ptr<StageImpl>> liveoutsFromStages(std::vector<Stage>& stages) {
            std::vector<std::shared_ptr<StageImpl>> impls;
            for (auto stage : stages) {
                impls.push_back(AstNodeImpl::asStageImpl(stage.impl()));
            }

            return liveoutsFromStages(impls);
        }

        //Get all liveouts coming out from a set of stages
        std::set<std::shared_ptr<StageImpl>> liveoutsFromStages(std::vector<std::shared_ptr<StageImpl>>& stages) {
            std::set<std::shared_ptr<StageImpl>> liveouts;
            std::set<PipelineStage*> allChildrenOfStages;
            for (auto stage : stages) {
                auto children = dslStageToPipelineStage[stage]->children();
                for (auto child : children) {
                    for (auto childStage : child->stages()) {
                        if (std::find(stages.begin(), stages.end(), childStage) == stages.end())
                            allChildrenOfStages.insert(children.begin(), children.end());
                    }
                }
            }

            for (auto stage : stages) {
                PipelineStage* ps = dslStageToPipelineStage[stage];
                auto psLiveouts = ps->liveoutStages(stageOutputs_);
                //Only add those liveouts that are outside the group of stages or in the output
                for (auto psLiveout : psLiveouts) {
                    if (std::find(outputs_.begin(), outputs_.end(), 
                                  dslStageToPipelineStage[psLiveout]) != outputs_.end())
                        liveouts.insert(psLiveout);
                    else if (allChildrenOfStages.count(dslStageToPipelineStage[psLiveout]) > 0) 
                        liveouts.insert(psLiveout);
                }
            }

            return liveouts;
        }

        void print(std::ostream& os) {
            ASSERT(topoOrder_.size() > 0, "Create DAG before printing.");
            os << "Pipeline: " << name() << ":" << std::endl;
            for (auto ps : topoOrder_) {
                SingleStagePrintVisitor printer(os);
                ps->print(os);
            }
        }

        void setAllStageStoreLoc();
        void codegen(std::string filename) 
        {codegen(filename, {});}
        void codegen(std::string filename, std::vector<ACCCDSLImpl::CodeGenVarBounds> varBounds) 
        {
            std::ofstream outputFile;
            outputFile.open(filename);
            if (outputFile.is_open()) {
                codegen(outputFile, varBounds);
                outputFile.close();
            } else {
                std::cout << "Error opening file: " << filename << std::endl;
            }
        }

        void codegen(std::ostream& os, std::vector<ACCCDSLImpl::CodeGenVarBounds> varBounds);

        Pipeline(const Pipeline& pipeline);
        ~Pipeline() 
        {
            //TODO: delete everything here
        }
    };

    class Autotuner 
    {        
        public:
        Autotuner() {}

        PipelineStage* findStageWithDefinition(Pipeline& pipeline, AstNodeType defType);
        bool isPointwiseStage(PipelineStage* ps);
        void autotune(Pipeline& pipeline) {autotune(pipeline, std::vector<Tensor>(0, Tensor(nullptr)));}
        void autotune(Pipeline& pipeline, std::vector<Tensor> canSlice);
    };
}

#endif
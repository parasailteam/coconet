#include "pipeline.hpp"
#include "codegen.hpp"

#include <stack>
#include <set>
#include <map>

using namespace ACCCDSL;
using namespace ACCCDSLImpl;

bool ACCCDSLImpl::ExpressionImpl::isPointwise()
{return isConstant() || (dims() == 1 && size(0)->isConstant() && AstNodeImpl::asConstInt32(size(0))->val() == 1);}

void ACCCDSLImpl::BinaryPointwiseOp::setupAndCheckDimensions()
{
    /*Following are the dimension and gpu checking rules
        *  ExpressionType ->       |    Constant           |   Variable          | Continuous Expression |   Scattered Expression
        *    |                     |                       |                     |                       |
        *   \ /                    |                       |                     |                       |
        *   Constant               |    Always legal       |  Dims and Layout      | Dims and Layout of      |  Dims and Layout of
        *                          |                       |  of Variable        | Expression            |     Expression
        * -------------------------------------------------------------------------------------------------------------
        *   Variable               |    Dimension and layout  |  Dims and layout      | Dims of Expression    |   Dims of Expression
        *                          |     of Variable       |  should match       | layout should match     |   layout should match
        * ----------------------------------------------------------------------------------------------------------------
        *   Continuous Expression  |    Dims and layout of    |  Dims of Expression | Dims and layout of      |   Illegal
        *                          |     Expression        |  layout should match  | Expressions           |
        *                          |                       |                     | should match          |
        * ----------------------------------------------------------------------------------------------------------------
        *   Scattered Expression   |    Dims and layout of    |  Dims of Expression |    Illegal            |  Dims and layout should
        *                          |    Expression         |  layout should match  |                       |   match
        * 
        */ 
    dimSizes_.clear();
    
    if (operand(0)->elemType() != operand(1)->elemType()) {
        ASSERT(false, "Type not same: '"<< TensorElemTypeToStr(operand(0)->elemType()) <<"' != '" << TensorElemTypeToStr(operand(1)->elemType()) << "'");
    }
    elemType_ = operand(0)->elemType();
    // if ((operand(0)->scattered() && (!operand(1)->isPointwise() && !operand(1)->scattered())) ||
    //     (operand(1)->scattered() && (!operand(0)->isPointwise() && !operand(0)->scattered())))
    //     ASSERT(false, "Operation between a scattered and continuous operand not allowed");
    
    size_t dimsOp1 = operand(0)->dims();
    size_t dimsOp2 = operand(1)->dims();
    
    size_t broadcastDim = std::max(dimsOp1, dimsOp2) - std::min(dimsOp1, dimsOp2); 
    static std::shared_ptr<ConstInt32> one(new ConstInt32(1));

    for (size_t dim = 0; dim < std::max(dimsOp1, dimsOp2); dim++) {
        std::shared_ptr<ExpressionImpl> sizeOp1;
        std::shared_ptr<ExpressionImpl> sizeOp2;

        if (dimsOp1 > dimsOp2) {
            sizeOp1 = operand(0)->isPointwise() ? one : operand(0)->size(dim);
            
            if (dim < broadcastDim) {
                sizeOp2 = sizeOp1;
            } else {
                sizeOp2 = operand(1)->isPointwise() ? one : operand(1)->size(dim);
            }
        } else if (dimsOp2 > dimsOp1) {
            sizeOp2 = operand(1)->isPointwise() ? one : operand(1)->size(dim);
            
            if (dim < broadcastDim) {
                sizeOp1 = sizeOp2;
            } else {
                sizeOp1 = operand(0)->isPointwise() ? one : operand(0)->size(dim);
            }
        } else {
            if (operand(0)->isPointwise() && operand(1)->isPointwise()) {
                sizeOp1 = one;
                sizeOp2 = one;
            } else if (operand(0)->isPointwise()) {
                sizeOp2 = operand(1)->size(dim);
                sizeOp1 = sizeOp2;
            } else if (operand(1)->isPointwise()) {
                sizeOp1 = operand(0)->size(dim);
                sizeOp2 = sizeOp1;
            } else {
                sizeOp1 = operand(0)->size(dim);
                sizeOp2 = operand(1)->size(dim);
            }
        }

        if (sizeOp1 != sizeOp2) {
            ASSERT(false, "First operand dims for dim '"<< dim <<"', '"<< sizeOp1->name() <<"' do not match with second '"<< sizeOp2->name() <<"'");
        }
        dimSizes_.push_back(sizeOp2);
    }

    //TODO: Set the group 
    ASSERT(false, "Set group");
    
    TensorLayout layoutOp1 = operand(0)->layout();
    TensorLayout layoutOp2 = operand(1)->layout();
    
    if (operand(0)->isPointwise() && operand(1)->isPointwise()) {
        layout_ = Replicated;
    } else if (operand(0)->isPointwise() || operand(1)->isPointwise()) {
        if (operand(0)->isPointwise()) {
            switch (layoutOp1) {
                case Local:
                    layout_ = Local;
                    break;
                case Replicated:
                    layout_ = layoutOp2;
                    break;
                case Sliced:
                    ASSERT(false, "Pointwise cannot be Sliced");
                    break;
            }
        } else {
            switch (layoutOp2) {
                case Local:
                    layout_ = Local;
                    break;
                case Replicated:
                    layout_ = layoutOp1;
                    break;
                case Sliced:
                    ASSERT(false, "Pointwise cannot be Sliced");
                    break;
            }
        }
    } else {
        //TODO: this needs more work based on the dimSizes() and layout
        if (layoutOp1 == layoutOp2) {
            layout_ = layoutOp1;
        } else if (layoutOp1 == Replicated) {
            layout_ = layoutOp2;
        } else if (layoutOp2 == Replicated) {
            layout_ = layoutOp1;
        } else {
            ASSERT(false, "First operand layout '"<<layoutOp1<<"' and second '"<<layoutOp2<<"' are invalid combination");
        }
    }
}

int AstNodeImpl::nameCounter = 0;

void Pipeline::codegen(std::ostream& os, std::vector<ACCCDSLImpl::CodeGenVarBounds> varBounds)
{
    //Create a map of Ast pointers to shared_ptr of Ast pointers
    for (auto iter : dslStageToPipelineStage) {
        std::queue<std::shared_ptr<AstNodeImpl>> q;
        q.push(iter.first);

        while (!q.empty()) {
            auto node = q.front();
            q.pop();

            astPtrToSharedPtr[node.get()] = node;
            for (auto child : node->children()) {
                if (child->type() != StageNode)
                    q.push(child);
            }
        }
    }
    NCCLCodegen codegen(*this, os);
    codegen.codegen(varBounds);
}

std::vector<Stage> stageImplsToStages(std::vector<std::shared_ptr<StageImpl>>& stageImpls)
{
    std::vector<Stage> stages;
    for (auto impl : stageImpls)
        stages.push_back(Stage(impl));
    return stages;
}

template<typename T, typename U>
std::vector<std::shared_ptr<U>> dslToAstImpls(std::vector<T> stages)
{
    std::vector<std::shared_ptr<U>> impls;
    for (auto stage : stages)
        impls.push_back(stage.impl());
    return impls;
}

template<class T>
void topologicalSortUtil(T* v, std::set<T*>& visited,  std::stack<T*>& stack,
                         int level) 
{ 
    // Mark the current node as visited. 
    visited.insert(v);
    
    // Recur for all the vertices  
    // adjacent to this vertex 
    for (auto iter : v->children()) {
        if (visited.find(iter) == visited.end()) 
            topologicalSortUtil(iter, visited, stack, level+1); 
    }
  
    // Push current vertex to stack  
    // which stores result
    stack.push(v); 
} 

void Pipeline::createTopologicalSort()
{
    std::stack<PipelineStage*> stack; 
    std::set<PipelineStage*> visited;

    topoOrder_.clear();
    for (auto iter : inputs()) {
        PipelineStage* ps = iter;
        if (visited.find(ps) == visited.end()) 
            topologicalSortUtil(ps, visited, stack, 0); 
    }

    while (!stack.empty()) {
        topoOrder_.push_back(stack.top());
        stack.top()->setTopoOrder(dslStageToPipelineStage.size() - stack.size());
        stack.pop();
    }
}

bool isCyclicUtil(PipelineStage* pipeStage, std::set<PipelineStage*>& visited, 
                  std::map<PipelineStage*, bool>& recStack) 
{ 
    if(visited.find(pipeStage) == visited.end()) 
    { 
        visited.insert(pipeStage); 
        recStack[pipeStage] = true;
  
        // Recur for all the vertices adjacent to this vertex 
        for(auto child : pipeStage->children()) 
        {
            bool isFound = (visited.find(child) != visited.end());
            if (!isFound && isCyclicUtil(child, visited, recStack) ) 
                return true; 
            else if (recStack[child]) 
                return true; 
        } 
  
    } 
    
    recStack[pipeStage] = false;
    return false; 
} 

bool isCyclic(Pipeline* pipeline) 
{ 
    std::set<PipelineStage*> visited;
    std::map<PipelineStage*, bool> recStack;

    for(auto stage : pipeline->topoOrder()) 
    { 
        recStack[stage] = false; 
    } 
  
    for(auto stage : pipeline->topoOrder()) 
        if (isCyclicUtil(stage, visited, recStack)) 
            return true; 
  
    return false; 
}

void Pipeline::setAllStageStoreLoc()
{
    for (auto it : dslStageToPipelineStage) {
        PipelineStage* pipeStage = it.second;

        pipeStage->setStorageLocation(stageOutputs_);
    }
}

struct TopoOrderComparer {
    bool operator()(const PipelineStage* ps1, const PipelineStage* ps2) const {
        return ps1->getTopoOrder() <= ps2->getTopoOrder();
    }
};

PipelineStage* Pipeline::combineStagesInDAG(std::vector<std::shared_ptr<StageImpl>> stagesToCombine, PipelineStageType combinationType)
{
    PipelineStage* firstTopoOrderStage = nullptr;
    std::unordered_set<std::shared_ptr<StageImpl>> uniqueStages;

    for (auto stageImpl : stagesToCombine) {
        bool found = false;
        uniqueStages.insert(stageImpl);
        found = (dslStageToPipelineStage.find(stageImpl) != dslStageToPipelineStage.end());
        ASSERT(found, "Given stage is not in the computation DAG");
    }

    //Find stage appearing earliest in the topological order
    for (auto pipeStage : topoOrder()) {
        for (auto stageImpl : uniqueStages) {
            PipelineStage* ps = dslStageToPipelineStage[stageImpl];
            if (ps == pipeStage) {
                firstTopoOrderStage = ps;
                break;
            }
        }

        if (firstTopoOrderStage) 
            break;
    }

    ASSERT(firstTopoOrderStage != nullptr, "First topological order stage not found");
    /*Remove all other Pipeline Stages and add their StageImpl to firstTopoOrderStage.*/

    //Get the pipeline stages in topological order.
    std::set<PipelineStage*, TopoOrderComparer> topoOrderOfFuseStages;
    std::unordered_set<PipelineStage*> uniquePipelineStages;

    for (auto stageImpl : uniqueStages) {
        PipelineStage* ps = dslStageToPipelineStage[stageImpl];
        if (uniquePipelineStages.count(ps) == 0) {
            uniquePipelineStages.insert(ps);
            topoOrderOfFuseStages.insert(ps);
        }
    }

    //Iterate all stages in topological order
    //Add all children and parents of ps2 to ps1.
    for (auto ps : topoOrderOfFuseStages) {
        if (ps == firstTopoOrderStage)
            continue;
        
        //Add ps children to firstTopoOrderStage's children
        for (auto child : ps->children()) {
            if (child != firstTopoOrderStage)
                firstTopoOrderStage->addChild(child);
        }

        //Add firstTopoOrderStage as parent to ps's children
        //and remove ps as parent of ps's children
        for (auto child : ps->children()) {
            if (child != firstTopoOrderStage)
                child->replaceParent(ps, firstTopoOrderStage);
            else 
                //If parent and firstTopoOrderStage are same
                //then just remove parent
                child->removeParent(ps);
        }

        //Add ps parent to firstTopoOrderStage's parent
        for (auto parent : ps->parents()) {
            if (firstTopoOrderStage != parent)
                firstTopoOrderStage->addParent(parent);
        }

        //Add firstTopoOrderStage as parent's child
        //and remove ps as child of ps's parent
        for (auto parent : ps->parents()) {
            if (firstTopoOrderStage != parent)
                parent->replaceChild(ps, firstTopoOrderStage);
            else
                //If parent and firstTopoOrderStage are same
                //then just remove parent
                parent->removeChild(ps);
        }

        for (auto s : ps->stages()) {
            dslStageToPipelineStage[s] = firstTopoOrderStage;
        }

        //Remove from topo order. We can safely do that because
        //firstTopoOrderStage is the earliest stage in topo order
        //and we are not removing it.
        auto it = std::find(topoOrder_.begin(), topoOrder_.end(), ps);
        ASSERT(it != topoOrder_.end(), "OOPS for "<<ps->stages()[0]->name());
        topoOrder_.erase(it);
        
        //Remove from inputs()
        auto findInInput = std::find(inputs_.begin(), inputs_.end(), ps);
        if (findInInput != inputs_.end()) {
            inputs_.erase(findInInput);
        }

        //Update outputs
        auto iter = std::find(outputs_.begin(), outputs_.end(), ps);
        if (iter != outputs_.end()) {
            outputs_.erase(iter);
            if (std::find(outputs_.begin(), outputs_.end(), firstTopoOrderStage) == outputs_.end())
                outputs_.push_back(firstTopoOrderStage);
        }
        //Update internal fused/overlap DAG
        firstTopoOrderStage->fuseOrOverlapStages(ps, combinationType);

        //Finally add stageImpl to firstTopoOrderStage
        for (auto s : ps->stages())
            firstTopoOrderStage->addStage(s);

        delete ps;
    }

    //Check for validity of fusion
    bool cyclic = isCyclic(this);

    if (cyclic == true)
        ASSERT(cyclic, "Fusion is not valid because it forms a cycle in DAG");

    //TODO: check only one comm collective within a fused stage
    //TODO: check that both s and t are of same size and are on same gpu

    //Recreate topological order
    createTopologicalSort();

    return firstTopoOrderStage;
}

bool Pipeline::checkIfAllStagesAreFused(std::vector<Stage> stages) 
{
    //Check if all fused stages are specified
    //TODO:
    return true;
}

void Pipeline::overlap(std::vector<Stage> stages) 
{
    ASSERT(checkIfAllStagesAreFused(stages), "Valid only when all stages of a fused stage are specified.");
    std::vector<std::shared_ptr<StageImpl>> stageImpls;
    for (auto stage : stages) {
        stageImpls.push_back(stage.impl());
    }
    overlap(stageImpls);
}

void Pipeline::fuse(std::vector<Stage> stagesToFuse) 
{
    ASSERT(checkIfAllStagesAreFused(stagesToFuse), "Valid only when all stages of a fused stage are specified.");
    std::vector<std::shared_ptr<StageImpl>> stageImpls;
    for (auto stage : stagesToFuse) {
        stageImpls.push_back(stage.impl());
    }
    fuse(stageImpls);
}

void Pipeline::fuse(std::vector<std::shared_ptr<StageImpl>> stagesToFuse)
{
    PipelineStage* ps = combineStagesInDAG(stagesToFuse, Fused);
    ps->setType(Fused);
}
void Pipeline::overlap(std::vector<std::shared_ptr<StageImpl>> stagesToOverlap)
{
    PipelineStage* ps = combineStagesInDAG(stagesToOverlap, Overlapped);
    ps->setType(Overlapped);
}

void Pipeline::asSlice(std::vector<Tensor> replicatedInputs)
{
    asSlice(dslToAstImpls<Tensor, TensorImpl>(replicatedInputs));
}

void Pipeline::asSlice(std::vector<std::shared_ptr<TensorImpl>> replicatedInputs)
{
    for (auto replicatedInput : replicatedInputs) {
        ASSERT(replicatedInput->layout() == Replicated, "asSlice works on only replicated tensors");

        //Find the Stage with Update expression for this Tensor
        // std::shared_ptr<UpdateImpl> update;
        // for (auto iter : dslStageToPipelineStage) {
        //     if (iter.first->definition()->type() == AstNodeType::UpdateNode) {
        //         update = AstNodeImpl::asUpdateImpl(iter.first->definition());
        //         if (update->arg() == replicatedInput.impl()) {
        //             break;
        //         }
        //     }
        // }

        std::vector<std::pair<PipelineStage*, std::shared_ptr<ExpressionImpl>>> psAndExprs;
        std::shared_ptr<StageImpl> extraAllGather = nullptr;
        std::shared_ptr<StageImpl> updateStage;

        for (auto ps : topoOrder_) {
            if (ps->usedExprsAsSharedPtr().count(replicatedInput)) {
                //Find all parent AST nodes of this Tensor
                std::stack<std::shared_ptr<AstNodeImpl>> exprStack;
                for (auto stage : ps->stages()) {
                    for (auto child : stage->children())
                        exprStack.push(child);

                    while (!exprStack.empty()) {
                        auto expr = exprStack.top();
                        exprStack.pop();
                        if (expr->type() == UpdateNode && 
                            AstNodeImpl::asUpdateImpl(expr)->arg() == replicatedInput) {
                            //Record the stage which updates the replicatedInput 
                            updateStage = stage;
                        }
                        for (auto child : expr->children()) {
                            if (expr->type() == UpdateNode && child == replicatedInput) {
                                continue;
                            } else if (child == replicatedInput) {
                                /*This transformation is valid only when this Tensor is used in a Slice format everywhere*/
                                ASSERT(expr->type() == ScatterNode, "Tensor should be used in only Slice layout");
                                psAndExprs.push_back(std::make_pair(ps, AstNodeImpl::asExpressionImpl(expr)));
                            }

                            if (expr->type() != StageNode)
                                exprStack.push(child);
                        }
                    }
                }
            }
        }

        //Replace Slice with tensor
        for (auto psAndExpr : psAndExprs) {
            psAndExpr.first->replaceExpr(psAndExpr.second, replicatedInput);
        }

        for (auto iter : dslStageToPipelineStage) {
            auto stage = iter.first;
            if (stage->definition()->type() == AllGatherNode) {
                if (stage->definition()->children()[0] == updateStage) {
                    extraAllGather = stage; 
                    break;
                }
            }
        }

        if (extraAllGather != nullptr) {
            PipelineStage* ps = dslStageToPipelineStage[extraAllGather];
            ps->removeStage(extraAllGather);

            //Replace this AllGather stage with its argument in the outputs
            auto iter = std::find(outputs_.begin(), outputs_.end(), ps);
            if (iter != outputs_.end()) {
                ASSERT(extraAllGather->definition()->children()[0]->type() == StageNode, "Input of AllGather is not a Stage.");
                auto updateStage = AstNodeImpl::asStageImpl(extraAllGather->definition()->children()[0]);
                ASSERT(stageOutputs_.count(extraAllGather) > 0, "Bug: Extra AllGather is not in stage outputs.");
                stageOutputs_.erase(extraAllGather);
                stageOutputs_.insert(updateStage);

                bool canRemovePS = true;
                //Remove this pipeline stage of AllGather only if none of its other stage are in the outputs
                for (auto stage : ps->stages()) {
                    if (stage != extraAllGather && 
                    std::find(stageOutputs_.begin(), stageOutputs_.end(), stage) != stageOutputs_.end()) {
                        canRemovePS = false;
                        break;
                    }
                }
                if (canRemovePS)
                    outputs_.erase(iter);

                if (std::find(outputs_.begin(), outputs_.end(), dslStageToPipelineStage[updateStage]) == outputs_.end())
                    outputs_.push_back(dslStageToPipelineStage[updateStage]);
            }

            if (ps->children().empty() && iter != outputs_.end()) {
                //Remove this stage if it is not being used later on
                std::set<PipelineStage*> children(ps->children());
                std::set<PipelineStage*> parents(ps->parents());

                for (auto child : children) {
                    child->addParents(parents);
                    child->removeParent(ps);
                }
                for (auto parent : parents) {
                    parent->addChildren(children);
                    parent->removeChild(ps);
                }
            }
        }

        //Recreate Topological Order
        createTopologicalSort();
    }
}

void Pipeline::updateExplicitStorageLocations()
{
    //Set storage locations for each stage
    for (auto iter : dslStageToPipelineStage) {
        if (iter.first->definition()->type() == UpdateNode) {
            //Store the stage in the Tensor node of Update
            explicitStoreLocations_[iter.first] = AstNodeImpl::asUpdateImpl(iter.first->definition())->arg();
        } else if (stageOutputs_.count(iter.first) == 0) {
            std::shared_ptr<ExpressionImpl> stageDef = iter.first->definition();                
            //If not in output then reuse the storage of the input
            switch (stageDef->type()) {
                case AllReduceNode: {
                    std::shared_ptr<AllReduceImpl> allReduceColl = AstNodeImpl::asAllReduceImpl(stageDef);
                    explicitStoreLocations_[iter.first] = allReduceColl->arg();
                    break;
                }
                case ReduceScatterNode: {
                    std::shared_ptr<ReduceScatterImpl> reduceScatterColl = AstNodeImpl::asReduceScatterImpl(stageDef);
                    explicitStoreLocations_[iter.first] = reduceScatterColl->arg();
                    break;
                }
                case AllGatherNode: {
                    std::shared_ptr<AllGatherImpl> allGatherColl = AstNodeImpl::asAllGatherImpl(stageDef);
                    explicitStoreLocations_[iter.first] = allGatherColl->arg();
                    break;
                }
            }
        }
    }
}

void Pipeline::fuseInto(std::vector<Expression> stages, CollCommOperationType t)
{
   /* Rules for fusion of comm collectives:
    * ReduceScatter and AllGather into AllReduce.
    * Reduce and Broadcast into AllReduce.
   **/
   
   //Check that all stages are in an already fused PipelineStage.
   
   PipelineStage* ps = dslStageToPipelineStage[AstNodeImpl::asStageImpl(stages[0].impl())];
   for (auto stage : stages) {
       PipelineStage* _ps = dslStageToPipelineStage[AstNodeImpl::asStageImpl(stage.impl())];
       if (_ps != ps) {
           ASSERT(false, "Not all stages has been fused into the same PipelineStage.");
           return;
       }
   }

   //TODO: Check that all stages can be fused into AllReduce
   //For now lets assume it can be 
   ps->fusedIntoCollComm(t);
}

AstNodeType stageDefinitionType(Stage stage)
{
    auto stageImpl = stage.impl();
    return stageDefinitionType(stageImpl);
}

AstNodeType stageDefinitionType(std::shared_ptr<StageImpl> stageImpl)
{
    return stageImpl->definition()->type();
}

ReorderedStages Pipeline::reorder(std::vector<Stage> comps, Stage allGather)
{
    auto ret = reorder(dslToAstImpls<Stage, StageImpl>(comps), allGather.impl());

    return ReorderedStages{stageImplsToStages(ret.first), stageImplsToStages(ret.second)};
}
 
std::pair<std::vector<std::shared_ptr<StageImpl>>, std::vector<std::shared_ptr<StageImpl>>>
Pipeline::reorder(std::vector<std::shared_ptr<StageImpl>> comps, std::shared_ptr<StageImpl> allGather)
{
    ASSERT(stageDefinitionType(allGather) == AllGatherNode, "reorder requires AllGather Stage");
    // ASSERT(checkIfAllStagesAreFused(comps), "Valid only when all stages of a fused stage are specified.");
    //TODO: Check that reorder of comps and allGather must include all computations in between them in DAG
    auto allGatherPS = dslStageToPipelineStage[allGather];
    std::vector<PipelineStage*> compsPS;
    for (auto comp : comps) 
        compsPS.push_back(dslStageToPipelineStage[comp]);
    
    std::vector<PipelineStage*> agChildComps;
    //All comps that are a child of allgather
    for (auto compPS : compsPS) 
        if (allGatherPS->hasChild(compPS)) 
            agChildComps.push_back(compPS);
    
    ASSERT(agChildComps.size() >= 1, "reorder requires atleast one of computations to take output of AllGather as their input.");

    //Get all the computations uses a stage/tensor in replicated layout
    std::queue<PipelineStage*> stageQueue;
    std::unordered_set<PipelineStage*> visitedStages;

    for (auto compPS : agChildComps) {
        stageQueue.push(compPS);
    }

    std::unordered_map<std::shared_ptr<ExpressionImpl>, std::set<PipelineStage*>> replicatedUsedExprs;

    while (!stageQueue.empty()) {
        PipelineStage* comp = stageQueue.front();
        stageQueue.pop();
        if (std::find(compsPS.begin(), compsPS.end(), comp) == compsPS.end())
            continue;
        std::set<std::shared_ptr<ExpressionImpl>> usedExprs = comp->usedExprsAsSharedPtr();
        
        for (auto expr : usedExprs) {
            if (expr != allGather) {
                if (expr->layout() == Local) {
                    //Transformation is not valid if any stage/tensor has a Local layout.
                    ASSERT(false, "Reorder does not work when an expression has a local layout.");
                }

                if ((expr->type() == StageNode || expr->type() == TensorNode) && 
                    expr->layout() == Replicated) {
                    if (replicatedUsedExprs.count(expr) == 0) {
                        replicatedUsedExprs[expr] = std::set<PipelineStage*>();
                    }

                    replicatedUsedExprs[expr].insert(comp);
                } else {
                    //Do nothing
                }
            }
        }

        visitedStages.insert(comp);

        for (auto child : comp->children()) {
            if (visitedStages.find(comp) != visitedStages.end())
                stageQueue.push(child);
        }
    }

    //Create one sliced stage for only replicated Tensors
    //Further setupAndCheckDimensions will flow the sliced layout to following stages
    std::unordered_map<std::shared_ptr<ExpressionImpl>, std::shared_ptr<ScatterImpl>> slicedForExprs;
    std::vector<Stage> slicedStages;
    for (auto it : replicatedUsedExprs) {
        auto expr = it.first;
        if (expr->type() != TensorNode)
            continue;
        std::shared_ptr<ScatterImpl> sliced = std::shared_ptr<ScatterImpl>(new ScatterImpl(AstNodeImpl::asTensorImpl(expr)));
        slicedForExprs.insert(std::make_pair(it.first, sliced));
    }

    for (auto iter : replicatedUsedExprs) {
        //In the AST replace replicatedExprs with the corresponding slicedStage
        if (iter.first->type() != TensorNode)
            continue;
        for (auto ps : iter.second) {
            ps->replaceExpr(iter.first, slicedForExprs.at(iter.first));
        }
    }

    //Remove the AllGather stage from DAG
    if (std::find(inputs_.begin(), inputs_.end(), allGatherPS) != inputs_.end()) {
        //Add all children of AllGather as inputs 
        for (auto child : allGatherPS->children()) {
            inputs_.push_back(child);
        }
    }

    for (auto child : allGatherPS->children())
        for (auto parent : allGatherPS->parents()) {
            parent->addChild(child);
            child->addParent(parent);
        }

    for (auto child : allGatherPS->children())
        child->removeParent(allGatherPS);
    for (auto parent : allGatherPS->parents())
        parent->removeChild(allGatherPS);

    //Replace output of AllGather with input of AllGather in all comps
    auto allGatherImpl = AstNodeImpl::asAllGatherImpl(allGather->definition());

    for (auto compStageImpl : comps) {
        PipelineStage* ps = dslStageToPipelineStage[compStageImpl];
        if (ps != nullptr) {
            ps->replaceExpr(allGather, allGatherImpl->arg());
            ps->removeParent(allGatherPS);
        }
    }

    // //For an UpdateNode that has Replicated layout for its arg() and Sliced for its update().
    // //Add an AllGather on update() and use that to update arg()
    // for (auto comp : comps) {
    //     if (comp.impl())->definition()->type() == UpdateNode {
    //         auto updateNode = AstNodeImpl::asUpdateImpl(comp.impl())->definition();
    //         if (updateNode->arg()->layout() == Replicated && updateNode->update()->layout() == Sliced) {

    //         }
    //     }
    // }

    //Setup dimensions and perform checks for all stages in comps again
    for (auto comp : comps) {
        PipelineStage* ps = dslStageToPipelineStage[comp];
        if (ps != nullptr)
            ps->setupAndCheckDimensions();
    }

    std::vector<Stage> newAllGatherStages;

    //Get all liveouts from comps
    std::set<std::shared_ptr<StageImpl>> liveouts = liveoutsFromStages(comps);
    std::vector<PipelineStage*> origOutputs = std::vector<PipelineStage*>(outputs_);
    // //For all liveouts that are Replicated generate an AllGather for them
    for (auto liveout : liveouts) {
        //Liveout will now be in Sliced layout after setupAndCheckDimensions
        if (liveout->layout() != Sliced)
            continue;
        
        Stage newAllGather = Stage(AllGather_(liveout));
        newAllGatherStages.push_back(newAllGather);
        PipelineStage* newAllGatherPS = createOrGetPipelineStageForDslStage(newAllGather.impl());
        
        //Add new AllGather stages after comps in DAG
        PipelineStage* liveoutPS = dslStageToPipelineStage[liveout];

        if (std::find(origOutputs.begin(), origOutputs.end(), liveoutPS) != origOutputs.end()) {
            //Handles cases when multiple liveouts are in same pipeline stage due to fusion and overlap
            auto liveoutPSIt = std::find(outputs_.begin(), outputs_.end(), liveoutPS);
            if (liveoutPSIt != outputs_.end()) {
                outputs_.erase(liveoutPSIt);
            }
            outputs_.push_back(newAllGatherPS);
            stageOutputs_.erase(liveout);
            stageOutputs_.insert(newAllGather.impl());
        }

        //For all Stages outside of reordered comps replace liveouts with their AllGather
        for (auto child : std::set<PipelineStage*>(liveoutPS->children())) {
            if (std::find(compsPS.begin(), compsPS.end(), child) != compsPS.end())
                continue;
            newAllGatherPS->addChild(child);
            liveoutPS->removeChild(child);
            child->replaceParent(liveoutPS, newAllGatherPS);
            child->replaceExpr(liveout, newAllGather.impl());
        }
        
        newAllGatherPS->addParent(liveoutPS);
        liveoutPS->addChild(newAllGatherPS);
    }

    //FIXME: Add existing AllGather if it's output is a liveout.
    //Recreate Topological Order
    createTopologicalSort();

    
    return std::make_pair(comps, dslToAstImpls<Stage, StageImpl>(newAllGatherStages));
}

std::pair<Stage, Stage> Pipeline::split(Stage allReduceStage, SplitType splitType)
{
    auto ret = split(allReduceStage.impl(), splitType);
    return std::make_pair(Stage(ret.first), Stage(ret.second));
}

std::pair<std::shared_ptr<StageImpl>, std::shared_ptr<StageImpl>> Pipeline::split(std::shared_ptr<StageImpl> allReduceStage, SplitType splitType)
{    //TODO: Assuming we can only split AllReduce.
    auto stageImpl = allReduceStage;
    ASSERT(stageDefinitionType(allReduceStage) == AllReduceNode, "split only works on AllReduce");
    
    AllReduceImpl* allReduceImpl = AstNodeImpl::asAllReduceImpl(stageImpl->definition()).get();
    auto reduceOp = allReduceImpl->reduceOp();

    
    Tensor allReduceInputAsTensor = Tensor(AstNodeImpl::asTensorImpl(allReduceImpl->arg()));
    Stage allReduceInputAsStage = Stage(allReduceImpl->arg());
    Stage reduceScatter = ReduceScatter(reduceOp, (allReduceImpl->arg()->type() == TensorNode) ? 
                                                   allReduceInputAsTensor :  allReduceInputAsStage);
    Stage allGather = AllGather(reduceScatter);

    /*******Update AST:*********/
    
    //Adding argument of AllReduce as argument to ReduceScatter is done already
    //Replace uses of AllReduce stage with AllGather stage
    for (auto stage: topoOrder_) {
        stage->replaceExpr(allReduceStage, allGather.impl());
    }

    /*******Update DAG:*********/
    PipelineStage* allReducePS = dslStageToPipelineStage[stageImpl];
    PipelineStage* reduceScatterPS = createOrGetPipelineStageForDslStage(reduceScatter.impl());
    PipelineStage* allGatherPS = createOrGetPipelineStageForDslStage(allGather.impl());

    //Parents of ReduceScatter PipelineStage(PS) are parents of AllReduce PS
    reduceScatterPS->addParents(allReducePS->parents());
    //ReduceScatter is parent of AllGather
    allGatherPS->addParent(reduceScatterPS);
    reduceScatterPS->addChild(allGatherPS);
    //Children of AllReduce are now children of AllGather
    allGatherPS->addChildren(allReducePS->children());
    //Replace AllReduce as child of parents of AllReduce with ReduceScatter
    for (auto parent : allReducePS->parents()) {
        parent->replaceChild(allReducePS, reduceScatterPS);
    }
    //Similarly with child of AllReduce
    for (auto child : allReducePS->children()) {
        child->replaceParent(allReducePS, allGatherPS);
    }

    std::replace(inputs_.begin(), inputs_.end(), allReducePS, reduceScatterPS);
    std::replace(outputs_.begin(), outputs_.end(), allReducePS, allGatherPS);
    
    //Recreate topological order
    createTopologicalSort();

    return std::make_pair(reduceScatter.impl(), allGather.impl());
}


void Pipeline::createDAG() 
{
    std::queue<std::shared_ptr<StageImpl>> stageQueue;
    for(auto stage : stageOutputs_) {
        stageQueue.push(stage);
    }
    while (!stageQueue.empty()) {
        StageParentsVisitor parentsVisitor;
        std::shared_ptr<StageImpl> stage = stageQueue.front();
        stageQueue.pop();
        
        parentsVisitor.start(*stage);
        PipelineStage* pipeStage = createOrGetPipelineStageForDslStage(stage);
        auto parents = pipeStage->usedStages();
        addPipelineStageParents(pipeStage, parents.begin(), parents.end());
        for (auto parent : parents) {
            PipelineStage* parentPipeStage = createOrGetPipelineStageForDslStage(parent);
            parentPipeStage->addChild(pipeStage);
            stageQueue.push(parent);
        }
    }

    for (auto it : dslStageToPipelineStage) {
        if (it.second->isPipelineInput())
            inputs_.push_back(it.second);
        if(it.second->isPipelineOutput(stageOutputs_)) 
            outputs_.push_back(it.second);
    }

    createTopologicalSort();
}

static int pipelineNameCounter = 0;

Pipeline::Pipeline(const Pipeline& pipeline) :
    name_(pipeline.name_+"-" + std::to_string(pipelineNameCounter++))
{
    //Clone the AST and update stageOutputs_
    CloneAstVisitor cloneVisitor;
    for (auto out : pipeline.stageOutputs_) {
        cloneVisitor.visit(*out);
        auto out2 = cloneVisitor.origToCloneMap().at(out.get());
        ASSERT(out2->type() == StageNode, "Bug");
        stageOutputs_.insert(AstNodeImpl::asStageImpl(out2));
    }
    origToCloneMap_ = cloneVisitor.origToCloneMap();

    //Recreate all Pipeline Stages
    std::unordered_map<PipelineStage*, PipelineStage*> oldToNewPipelineStage;
    for (auto oldPS : pipeline.topoOrder_) {
        std::vector<std::shared_ptr<StageImpl>> newStages;
        for (auto oldStage : oldPS->stages()) {
            auto newStage = AstNodeImpl::asStageImpl(cloneVisitor.origToCloneMap().at(oldStage.get()));
            newStages.push_back(newStage);
        }
        auto newPS = new PipelineStage(newStages);
        newPS->type_ = oldPS->type_;
        for (auto newStage : newStages) {
            dslStageToPipelineStage[newStage] = newPS;
        }
        oldToNewPipelineStage[oldPS] = newPS;
    }
    
    //Recreate internal DAG of each pipeline stage
    for (auto iter : oldToNewPipelineStage) {
        if (iter.first->fuseOverlapDAG_ == nullptr)
            continue;
        
        std::shared_ptr<PipelineStage::FuseOverlapNode> node = iter.first->fuseOverlapDAG_;
        std::vector<std::shared_ptr<StageImpl>> newStages;
        for (auto oldStage : node->stages) {
            auto newStage = AstNodeImpl::asStageImpl(cloneVisitor.origToCloneMap().at(oldStage.get()));
            newStages.push_back(newStage);
        }
        iter.second->fuseOverlapDAG_ = std::shared_ptr<PipelineStage::FuseOverlapNode>(
                new PipelineStage::FuseOverlapNode{node->type, newStages, nullptr, nullptr});
        auto parent = iter.second->fuseOverlapDAG_;
        node = node->child;

        while (node != nullptr) {
            for (auto oldStage : node->stages) {
                auto newStage = AstNodeImpl::asStageImpl(cloneVisitor.origToCloneMap().at(oldStage.get()));
                newStages.push_back(newStage);
            }
            auto newNode = std::shared_ptr<PipelineStage::FuseOverlapNode>(
                new PipelineStage::FuseOverlapNode{node->type, newStages, parent, nullptr});
            parent->child = newNode;
            parent = newNode;
            node = node->child;
            // os << "}" << std::endl;
        }
    }
    //Assign new DAG children and parents
    for (auto iter : oldToNewPipelineStage) {
        for (auto child : iter.first->children())
            iter.second->addChild(oldToNewPipelineStage.at(child));
        for (auto parent : iter.first->parents())
            iter.second->addParent(oldToNewPipelineStage.at(parent));
    }

    for (auto oldOutput : pipeline.outputs_)
        outputs_.push_back(oldToNewPipelineStage.at(oldOutput));

    for (auto oldInput : pipeline.inputs_)
        inputs_.push_back(oldToNewPipelineStage.at(oldInput));

    for (auto oldArg : pipeline.arguments_) {
        try {
            auto newArg = cloneVisitor.origToCloneMap().at(oldArg.get());
            arguments_.push_back(AstNodeImpl::asExpressionImpl(newArg));
        } catch (...) {

        }
    }
    
    createTopologicalSort();
}   

bool Autotuner::isPointwiseStage(PipelineStage* ps)
{
    const std::set<AstNodeType> pointwiseTypes 
    {
        TensorNode,
        BinaryPointwiseOpNode,
        UnaryPointwiseOpNode,
        ReduceNode,
        ReduceTensorNode,
        StageNode,
        VariableNode,
        ConstantNode,
        CastNode,
        ScatterNode,
        IteNode
    };

    if (pointwiseTypes.count(ps->stages()[0]->definition()->type()) > 0) {
            return true;
    } else if (ps->stages()[0]->definition()->type() == UpdateNode) {
        if (pointwiseTypes.count(AstNodeImpl::asUpdateImpl(ps->stages()[0]->definition())->update()->type()) > 0) {
            return true;
        }
    }

    return false;
}

PipelineStage* Autotuner::findStageWithDefinition(Pipeline& pipeline, AstNodeType defType)
{
    for (auto ps : pipeline.topoOrder()) {
        //A stage can contain a BinaryPointwise node, an Update Node, or a Unary node
        if (ps->stages()[0]->definition()->type() == defType)
            return ps;
    }

    return nullptr;
}

void Autotuner::autotune(Pipeline& origPipeline, std::vector<Tensor> canSlice)
{
    //Find all contiguous pointwise operations
    for (auto ps : origPipeline.topoOrder()) {
        ASSERT(ps->stages().size() == 1, "Autotuner cannot work when there already some fused stages");
    }

    std::vector<PipelineStage*> pointwiseStages;

    for (auto ps : origPipeline.topoOrder()) {
        //A stage can contain a BinaryPointwise node, an Update Node, or a Unary node
        if (isPointwiseStage(ps)) 
            pointwiseStages.push_back(ps);
    }

    //Find sub-DAG starting from each pointwise stage
    std::vector<std::vector<PipelineStage*>> pointwiseSubDAGs;
    std::unordered_set<PipelineStage*> visited;
    //Go through the topological order in reverse
    for (auto psStart = pointwiseStages.rbegin(); psStart != pointwiseStages.rend(); psStart++) {
        if (visited.count(*psStart) == 1)
            continue;

        std::vector<PipelineStage*> subDAG;
        std::stack<PipelineStage*> psStack;
        psStack.push(*psStart);

        while (!psStack.empty()) {
            auto ps = psStack.top();
            psStack.pop();

            if (visited.count(ps) == 1)
                continue;

            visited.insert(ps);
            subDAG.push_back(ps);
            
            for (auto parent : ps->parents()) {
                if (isPointwiseStage(parent))
                    psStack.push(parent);
            }
        }

        if (subDAG.size() > 1)
            pointwiseSubDAGs.push_back(subDAG);
    }

    for (auto subDAG : pointwiseSubDAGs) {
        //Fuse all pointwise sub DAGs
        std::vector<std::shared_ptr<StageImpl>> impls;
        for (auto ps : subDAG) {
            //All pipeline stages are required to only have one AST stage
            impls.push_back(ps->stages()[0]);
        }
        origPipeline.fuse(impls);
    }

    //At this point, the original pipeline has all possible pointwise stages fused
    //Now do the tuning by going through all schedules
    //The while loop runs until there is no more transformation to do
    //Each iteration of the loop applies one or more transformation and generates a new Pipeline
    std::vector<Pipeline*> allPipelines;
    allPipelines.push_back(&origPipeline);
    std::vector<std::shared_ptr<TensorImpl>> canSliceTensorImpl = dslToAstImpls<Tensor, TensorImpl>(canSlice);
    bool change = true;
    while(change) {
        Pipeline* prevPipeline = allPipelines.back();
        auto allReduceStage = findStageWithDefinition(*prevPipeline, AllReduceNode);
        change = false;
        if (allReduceStage) {
            // An Allreduce exist so create a new pipeline and do transformations there
            Pipeline* currPipeline = new Pipeline(*prevPipeline);
            allReduceStage = findStageWithDefinition(*currPipeline, AllReduceNode);
            /*Split the All Reduce Stage*/
            auto rsAg = currPipeline->split(allReduceStage->stages()[0], SplitType::AllReduceRSAG);
            /*Since there is no point in just splitting AllReduce, lets reorder the computation with
            AllGather*/
            //Since computations are fused, so child of AllGather can be reordered
            PipelineStage* allGatherPS = currPipeline->dslStageToPipelineStage[rsAg.second];
            ASSERT(allGatherPS->children().size() == 1, "Current code assumes that AllGather has only one child");

            currPipeline->reorder((*allGatherPS->children().begin())->stages(), 
                                  allGatherPS->stages()[0]);
            std::vector<std::shared_ptr<TensorImpl>> currSliceTensorImpls;
            for (auto slice : canSliceTensorImpl) {
                currSliceTensorImpls.push_back(AstNodeImpl::asTensorImpl(currPipeline->origToCloneMap_.at(slice.get())));
            }
            //Slice the tensors
            if (!currSliceTensorImpls.empty())
                currPipeline->asSlice(currSliceTensorImpls);
            Pipeline* ww = new Pipeline(*currPipeline);
            allPipelines.push_back(currPipeline);
            change = true;
        } else {
            /*Find stages to fuse*/

            //Find a sequence of ReduceScatter, Computation, and AllGather stages and fuse them.
            auto allGatherStage = findStageWithDefinition(*prevPipeline, AllGatherNode);
            bool canFuse = false;
            std::vector<std::shared_ptr<StageImpl>> toFuse;
            if (allGatherStage) {
                //TODO: Assuming all stages are in a line now.
                PipelineStage* computationStage = *allGatherStage->parents().begin();
                if (computationStage) {
                    PipelineStage* reduceScatterStage = *computationStage->parents().begin();
                    canFuse = reduceScatterStage != nullptr && reduceScatterStage->stages()[0]->definition()->type() == ReduceScatterNode;
                }
            }

            if (canFuse) {
                Pipeline* currPipeline = new Pipeline(*prevPipeline);
                auto allGatherStage = findStageWithDefinition(*currPipeline, AllGatherNode);
                std::vector<std::shared_ptr<StageImpl>> toFuse;
                PipelineStage* computationStage = *allGatherStage->parents().begin();
                PipelineStage* reduceScatterStage = *computationStage->parents().begin();
                toFuse.push_back(reduceScatterStage->stages()[0]);
                toFuse.insert(toFuse.begin(), computationStage->stages().begin(), computationStage->stages().end());
                toFuse.push_back(allGatherStage->stages()[0]);

                currPipeline->fuse(toFuse);
                allPipelines.push_back(currPipeline);
                change = true;
            }
        }
    }

    for (auto pipe : allPipelines) {
        pipe->print(std::cout);
        std::cout << std::endl << std::endl;
    }
}

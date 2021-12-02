#include <gtest/gtest.h>

#include <dsl.hpp>
#include <ast.hpp>
#include <pipeline.hpp>

using namespace ACCCDSL;

TEST(FusionValidityTest, BaseCase) {
    const int N = 1024;
    const int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    
    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;

    Pipeline pipeline({g,w,lr}, {w1});
    ASSERT_TRUE(pipeline.fuse({w1, g1}));
    auto toporder = pipeline.topoOrder();
    ASSERT_TRUE(toporder.size() == 1);
    ASSERT_TRUE(toporder[0]->children().size() == 0);
    ASSERT_TRUE(toporder[0]->parents().size() == 0);
    StageImpl* g1impl = dynamic_cast<ACCCDSLImpl::StageImpl*>(g1.impl().get());
    StageImpl* w1impl = dynamic_cast<ACCCDSLImpl::StageImpl*>(w1.impl().get());
    ASSERT_TRUE(std::find(toporder[0]->stages().begin(),
                toporder[0]->stages().end(), g1impl) != toporder[0]->stages().end());
    ASSERT_TRUE(std::find(toporder[0]->stages().begin(),
                toporder[0]->stages().end(), w1impl) != toporder[0]->stages().end());
}

TEST(FusionValidityTest, FusingUnusedStages) {
    const int N = 1024;
    const int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor o(TensorElemType::Float32, N, NumGPUs, "o");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;
    Stage o1 = o - w;

    Pipeline pipeline({g,w,o,lr}, {o1});
    ASSERT_FALSE(pipeline.fuse({w1, g1}));
}

TEST(FusionValidityTest, FusingThreeStages) {
    const int N = 1024;
    const int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor o(TensorElemType::Float32, N, NumGPUs, "o");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;
    Stage o1 = o - w1;

    Pipeline pipeline({g,w,o,lr}, {o1});
    ASSERT_TRUE(pipeline.fuse({o1, w1, g1}));
}

TEST(FusionValidityTest, CyclesinFusion) {
    const int N = 1024;
    const int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor o(TensorElemType::Float32, N, NumGPUs, "o");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;
    Stage o1 = o - w1;

    Pipeline pipeline({g,w,o,lr}, {o1});
    ASSERT_FALSE(pipeline.fuse({o1, g1}));
}
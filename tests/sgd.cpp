#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>
//TODO: include<coconet.hpp>

using namespace ACCCDSL;

void sgd()
{
    int N = 1024;
    
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;
    
    Pipeline pipeline({g,w,lr}, {w1});
    pipeline.print(std::cout);
}

void sgdSplit() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - g1;
    
    Pipeline pipeline({g,w,lr}, {w1});
    
    auto rsAg = pipeline.split(g1, SplitType::AllReduceRSAG);
    pipeline.print(std::cout);
}

void sgdReorder() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = ReduceScatter(Summation, g);
    Stage g2 = AllGather(g1);
    Stage w1 = w - g2;
    
    Pipeline pipeline({g,w,lr}, {w1});
    
    pipeline.reorder({w1}, g2);
    pipeline.print(std::cout);
}

void sgdFusedAllReduce() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Replicated, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");
    
    Stage g1 = ReduceScatter(Summation, g);
    Stage ws = Scatter(w);
    Stage w1 = ws - g1;
    Stage w2 = AllGather(w1);

    Pipeline pipeline({g, w, lr}, {w2});
    // pipeline.fuse({g1, ws, w1, w2});
    pipeline.print(std::cout);
    //pipeline.codegen(std::cout);
}

#if 0

void sgd16() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float16, NumGPUs, "lr");
    Tensor g(TensorElemType::Float16, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w + Cast(TensorElemType::Float32, g1);

    // fuse(w1, g1);
    //store_at(w1, w);
    //store_at(m1, m);
    Pipeline pipeline({g,w}, {w1});

    pipeline.fuse({w1, g1});
    pipeline.storeAt(w1, w);
    pipeline.print(std::cout);
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
}

void sgd_reducescatter() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    
    ScatteredStage g1 = ReduceScatter(Summation, g);
    ScatteredStage ws = Scatter(w);
    ScatteredStage w1 = ws - lr * g1;
    Stage w2 = AllGather(w1);

    Pipeline pipeline({g, w, lr}, {w2});
    pipeline.print(std::cout);
    //pipeline.codegen(std::cout);
}
#endif

int main()
{
    //pos();
    //sgd2();
    // std::cout << "sgd-reducescatter:" << std::endl;
    // sgd_reducescatter();
    // std::cout << "adam " << std::endl;
    //adam();
    //adamf16ReduceScatterAllGather();
    //sgd16();
    // sgdReorder();
    //sgdFusedAllReduce();
    // lamb();
    //sgd_reduce();
    sgdSplit();
    return 0;    
}
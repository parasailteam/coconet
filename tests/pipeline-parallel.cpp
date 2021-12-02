#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

//nvcc test.cu -lcudadevrt -lcudart -I.. -I/usr/local/cuda/include/ -I ../../nccl-2/build/include/ -L../../nccl-2/build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && mpicxx test.o -I/usr/local/cuda/include/ -I../../nccl-2/build/include/ -L../../nccl-2/build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand -lcudadevrt

//TODO:
//Test cases using googletest
//Stage -> Var
//ScatteredStage -> Sliced
//Support FuseCollectives
//Support Overlap
//Support Transformations: asdead, asslice, split, reorder

using namespace ACCCDSL;
void sgd() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, "lr");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");

    Stage g1 = AllReduce(Summation, g);
    Stage w1 = w - lr * g1;
    
    Pipeline pipeline({g,w,lr}, {w1});

    pipeline.fuse({w1, g1});
    pipeline.storeAt(w1, w);
    pipeline.print(std::cout);
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
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
    pipeline.reorder({w1}, rsAg.second);
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

void adam() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Variable beta1(TensorElemType::Float32, NumGPUs, "beta1");
    Variable beta2(TensorElemType::Float32, NumGPUs, "beta2");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor m(TensorElemType::Float32, N, NumGPUs, "m");
    Tensor v(TensorElemType::Float32, N, NumGPUs, "v");
    
    Stage g1 = AllReduce(Summation, g);
    Stage m1 = beta1 * m + (1 - beta1) * g1;
    Stage v1 = beta2 * v + (1 - beta2) * (g1 * g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = w - lr * m_ / v_;
    
    /**
     * Rules for the storage and computation description:
     * 1. If a stage s is stored as scatterStore(s), then
     *    (0) its input storage requirement also becomes 1/(Number of GPUs)
     *    (i) If fused with a stage with comm collective:
     *         I. If fused with a AllReduce, then it is computed at the place
     *            where allreduced value is returned.
     *         II. If fused with Broadcast, then it is computed when values
     *             are received by each GPU.
     *         III. If fused with ReduceScatter, then it is computed when the
     *              values are received by each GPU, i.e., the scatter part.
     *         IV. If fused with Reduce, then it does not makes sense, because
     *              only GPU receives the Reduced values.
     *         V. If fused with AllGather, then it is computed before sending
     *            the value to the GPUs.
     *    (ii) its computation is parallelized across GPUs.
     *    (iii) And if s is already a ScatteredStage then it is okay.
     *    (iv) And if s is storeAt at some other location then it is legal
     *         only if the sizes match.
     *    (v) it can be fused with other stages only when they are scattered too.
     * 2. Each fused stage can only have one communication collective in it.
     * 3. Fusion should not lead to any cyclic dependencies in the resulting DAG.
     * 4. storeAt(s, t) is legal when t is not used anywhere after
     *    the computation of s.
    */

//Compile test.cu nvcc test.cu -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand && mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
    Pipeline pipeline({g, w, m, v, lr, beta1, beta2}, {w1});
    //Ideally, this should be illegal because m1 and v1 are not scattered 
    //but they are treated as scattered. So, for now we are going to allow 
    //code generation for this and later we will not do that.
    pipeline.fuse({g1, w1, m1, v1, m_, v_});
    pipeline.storeAt(w1, w);
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.print(std::cout);
    pipeline.codegen(std::cout, GenMultiProcessCode | GenMainFunction | GenResultsCheckCode);
}

void adamfMVG16() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float16, NumGPUs, "lr");
    Variable beta1(TensorElemType::Float16, NumGPUs, "beta1");
    Variable beta2(TensorElemType::Float16, NumGPUs, "beta2");
    Tensor g(TensorElemType::Float16, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor m(TensorElemType::Float16, N, NumGPUs, "m");
    Tensor v(TensorElemType::Float16, N, NumGPUs, "v");

    Stage g1 = AllReduce(Summation, g);
    Stage m1 = beta1 * m + (Const<float>(TensorElemType::Float16, 1) - beta1) * g1;
    Stage v1 = beta2 * v + (Const<float>(TensorElemType::Float16, 1) - beta2) * (g1 * g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = w - Cast(TensorElemType::Float32, lr * m_ / v_);
    
    //Compile test.cu nvcc test.cu -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand && mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
    Pipeline pipeline({g, w, m, v, lr, beta1, beta2}, {w1});
    //Ideally, this should be illegal because m1 and v1 are not scattered 
    //but they are treated as scattered. So, for now we are going to allow 
    //code generation for this and later we will not do that.
    pipeline.fuse({g1, w1, m1, v1, m_, v_});
    pipeline.storeAt(w1, w);
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.codegen(std::cout, GenMultiProcessCode | GenMainFunction | GenResultsCheckCode);
}

void adamfG16() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Variable beta1(TensorElemType::Float32, NumGPUs, "beta1");
    Variable beta2(TensorElemType::Float32, NumGPUs, "beta2");
    Tensor g(TensorElemType::Float16, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    Tensor m(TensorElemType::Float32, N, NumGPUs, "m");
    Tensor v(TensorElemType::Float32, N, NumGPUs, "v");

    Stage g1 = AllReduce(Summation, g);
    Stage m1 = beta1 * m + (1.0f - beta1) * Cast(TensorElemType::Float32, g1);
    Stage v1 = beta2 * v + (1.0f - beta2) * Cast(TensorElemType::Float32, g1 * g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = w - lr * m_ / v_;
    
    //Compile test.cu nvcc test.cu -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand && mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
    Pipeline pipeline({g, w, m, v, lr, beta1, beta2}, {w1});
    //Ideally, this should be illegal because m1 and v1 are not scattered 
    //but they are treated as scattered. So, for now we are going to allow 
    //code generation for this and later we will not do that.
    pipeline.fuse({g1, w1, m1, v1, m_, v_});
    pipeline.storeAt(w1, w);
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.codegen(std::cout, GenMultiProcessCode | GenMainFunction | GenResultsCheckCode);
}

void adamf16ReduceScatterAllGather() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Variable beta1(TensorElemType::Float32, NumGPUs, "beta1");
    Variable beta2(TensorElemType::Float32, NumGPUs, "beta2");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    ScatteredTensor m(TensorElemType::Float32, N, NumGPUs, "m");
    ScatteredTensor v(TensorElemType::Float32, N, NumGPUs, "v");

    ScatteredStage g1 = ReduceScatter(Summation, g);
    ScatteredStage m1 = beta1 * m + (1.0f - beta1) * g1;
    ScatteredStage v1 = beta2 * v + (1.0f - beta2) * g1 * g1;
    ScatteredStage m_ = m1/beta1;
    ScatteredStage v_ = v1/beta2;
    ScatteredStage scatteredUpdate = lr * m_ / v_;
    Stage update = AllGather(scatteredUpdate);
    Stage w1 = w - update;
    
    //Compile test.cu nvcc test.cu -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand && mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
    Pipeline pipeline({g, w, m, v, lr, beta1, beta2}, {w1});
    //Ideally, this should be illegal because m1 and v1 are not scattered 
    //but they are treated as scattered. So, for now we are going to allow 
    //code generation for this and later we will not do that.
    pipeline.print(std::cout);
    pipeline.fuse({g1, w1, m1, v1, m_, v_, scatteredUpdate, update, w1});
    pipeline.print(std::cout);
    pipeline.storeAt(w1, w);
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.fuseInto({g1, w1, m1, v1, m_, v_, scatteredUpdate, update}, AllReduceOp);
    pipeline.codegen(std::cout, GenMultiProcessCode | GenMainFunction | GenResultsCheckCode);
}

void lamb()
{
    int N = 1024;
    int P = 4;

    Variable lr(TensorElemType::Float32, P, "lr");
    Variable beta1(TensorElemType::Float32, P, "beta1");
    Variable beta2(TensorElemType::Float32, P, "beta2");
    Variable epsilon(TensorElemType::Float32, P, "epsilon");
    Variable gamma(TensorElemType::Float32, P, "gamma");
    Variable wNorm(TensorElemType::Float32, P, "w_norm");
    Tensor w(TensorElemType::Float32, N, P, "w");
    Tensor g(TensorElemType::Float32, N, P, "g");
    Tensor m(TensorElemType::Float32, N, P, "m");
    Tensor v(TensorElemType::Float32, N, P, "v");

    Stage g1 = AllReduce(Summation, g);
    Stage m1 = beta1 * m + (1.0 - beta1) * g1;
    Stage v1 = beta2 * v + (1.0 - beta2) * g1 * g1;
    Stage m_ = m1 / (1.0 - beta1);
    Stage v_ = v1 / (1.0 - beta2);
    Stage u = m_ / Sqrt(v_ + epsilon) + gamma * w;
    Stage uu = u*u;
    Stage r2Red = ReduceStage(uu, Summation);
    Stage r2 = Sqrt(r2Red);
    Stage r = Ite(wNorm > 0, Ite(r2 > 0, wNorm / r2, 1.0), 1.0);
    Stage w1 = w - lr * r * u;

    Pipeline pipeline({g, w, m, v, lr, beta1, beta2, epsilon, gamma, wNorm}, {w1});
    pipeline.fuse({g1,m1,v1,m_,v_,u,uu,r2Red,r2,r,w1});
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.storeAt(w1, w);
    //pipeline.storeAt(u, g); // TODO: this generates broken code currently
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
}
// TODO: this segfaults
// Stage r1 = Sqrt(ReduceStage(w*w, Summation));
// While this does not
// Stage ww = w*w;
// Stage r1 = Sqrt(ReduceStage(ww, Summation));

/*
void lambReduceScatter()
{
    int N = 1024;
    int P = 4;

    Variable lr(TensorElemType::Float32, P, "lr");
    Variable beta1(TensorElemType::Float32, P, "beta1");
    Variable beta2(TensorElemType::Float32, P, "beta2");
    Variable epsilon(TensorElemType::Float32, P, "epsilon");
    Variable gamma(TensorElemType::Float32, P, "gamma");
    Tensor w(TensorElemType::Float32, N, P, "w");
    Tensor g(TensorElemType::Float32, N, P, "g");
    ScatteredTensor m(TensorElemType::Float32, N, P, "m");
    ScatteredTensor v(TensorElemType::Float32, N, P, "v");

    ScatteredStage g1 = ReduceScatter(Summation, g);
    ScatteredStage m1 = beta1 * m + (1.0 - beta1) * g1;
    ScatteredStage v1 = beta2 * v + (1.0 - beta2) * g1 * g1;
    ScatteredStage m_ = m1 / (1.0 - beta1);
    ScatteredStage v_ = v1 / (1.0 - beta2);
    ScatteredStage u = m_ / Sqrt(v_ + epsilon) + gamma * w;
    ScatteredStage ww = w*w;
    ScatteredStage r1Red = ReduceStage(ww, Summation);
    ScatteredStage r1 = Sqrt(r1Red);
    ScatteredStage uu = u*u;
    ScatteredStage r2Red = ReduceStage(uu, Summation);
    ScatteredStage r2 = Sqrt(r2Red);
    ScatteredStage r = Ite(r1 > 0, Ite(r2 > 0, r1 / r2, 1.0), 1.0);
    ScatteredStage scatteredUpdate = lr * r * u;
    Stage update = AllGather(scatteredUpdate);
    Stage w1 = w - update;

    Pipeline pipeline({g, w, m, v, lr, beta1, beta2, epsilon, gamma}, {w1});
    pipeline.fuse({g1,m1,v1,m_,v_,u,ww,r1Red,r1,uu,r2Red,r2,r,scatteredUpdate,update,w1});
    pipeline.storeAt(m1, m);
    pipeline.storeAt(v1, v);
    pipeline.storeAt(w1, w);
    pipeline.fuseInto({g1,m1,v1,m_,v_,u,ww,r1Red,r1,uu,r2Red,r2,r,scatteredUpdate,update}, AllReduceOp);
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
}
*/

void pos() 
{
    size_t b = 64;
    size_t s = 512;
    size_t h = 1024;
    size_t size = b*s*h;
    int NumGPUs = 16;

    Tensor input(TensorElemType::Float16, size, NumGPUs, "input");
    Tensor bias(TensorElemType::Float16, size, NumGPUs, "bias");
    Tensor dropout(TensorElemType::Float16, size, NumGPUs, "dropout");
    Tensor resid(TensorElemType::Float16, size, NumGPUs, "resid");

    Stage s1 = AllReduce(Summation, input);
    Stage s2 = (s1 + bias) * dropout + resid;

    Pipeline pipeline({input, bias, dropout, resid}, {s2});
    pipeline.fuse({s1, s2});
    pipeline.storeAt(s2, resid);
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
}

void sgd_reduce() 
{
    int N = 1024;
    int NumGPUs = 4;
    
    Variable lr(TensorElemType::Float32, NumGPUs, "lr");
    Tensor g(TensorElemType::Float32, N, NumGPUs, "g");
    Tensor w(TensorElemType::Float32, N, NumGPUs, "w");
    
    Stage g1 = AllReduce(Summation, g);
    Stage g2 = w + Cast(Float32, g1);
    Stage r = ReduceStage(g2, Summation);
    Stage w1 = w - Cast(Float32, r * g1);

    Pipeline pipeline({g, w, lr}, {w1});
    pipeline.fuse({g1, g2, r, w1});
    pipeline.storeAt(w1, w);
    pipeline.print(std::cout);
    pipeline.codegen(std::cout, ACCCDSL::CodeGenOptions::GenMainFunction | ACCCDSL::CodeGenOptions::GenMultiProcessCode);
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
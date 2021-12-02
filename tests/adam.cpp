#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

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
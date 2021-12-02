#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>


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
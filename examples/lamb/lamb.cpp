#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

using namespace ACCCDSL;
//https://arxiv.org/pdf/1904.00962.pdf

void lambAR_C()
{
    Variable N(Int32, "N");
    Variable lr(Float32, "lr");
    Variable beta1(Float32, "beta1");
    Variable beta2(Float32, "beta2");
    Variable epsilon(Float32, "epsilon");
    Variable lambda(Float32, "gamma");
    Tensor w(Float32, N, Replicated, "w");
    Tensor g(Float32, N, Local, "g");
    Tensor m(Float32, N, Replicated, "m");
    Tensor v(Float32, N, Replicated, "v");

    Stage g1 = AllReduce(Summation, g);
    Stage m1 = Update(m, beta1 * m + (1.0 - beta1) * g1);
    Stage v1 = Update(v, beta2 * v + (1.0 - beta2) * g1 * g1);
    Stage m_ = m1 / (1.0 - beta1);
    Stage v_ = v1 / (1.0 - beta2);
    Stage r = m_ / Sqrt(v_) + lambda * w;
    Stage rNorm = Norm(r);
    Stage wNorm = Norm(w);
    Stage w1 = w - lr * (wNorm/rNorm) * r;

    Pipeline pipeline("lamb", {g, w, m, v, lr, beta1, beta2, epsilon, lambda}, {w1});
    pipeline.fuse({m1,v1,m_,v_,r,rNorm,wNorm,w1});
    // pipeline.print(std::cout);
    pipeline.codegen("lamb-ar-c.cu");
}


void lambRS_C_AG()
{
    Variable N(Int32, "N");
    Variable lr(Float32, "lr");
    Variable beta1(Float32, "beta1");
    Variable beta2(Float32, "beta2");
    Variable epsilon(Float32, "epsilon");
    Variable lambda(Float32, "gamma");
    Tensor w(Float32, N, Replicated, "w");
    Tensor g(Float32, N, Local, "g");
    Tensor m(Float32, N, Replicated, "m");
    Tensor v(Float32, N, Replicated, "v");

    Stage g1 = AllReduce(Summation, g);
    Stage m1 = Update(m, beta1 * m + (1.0 - beta1) * g1);
    Stage v1 = Update(v, beta2 * v + (1.0 - beta2) * g1 * g1);
    Stage m_ = m1 / (1.0 - beta1);
    Stage v_ = v1 / (1.0 - beta2);
    Stage r = m_ / Sqrt(v_) + lambda * w;
    Stage rNorm = Norm(r);
    Stage wNorm = Norm(w);
    Stage w1 = Update(w, w - lr * (wNorm/rNorm) * r);

    Pipeline pipeline("lamb", {g, w, m, v, lr, beta1, beta2, epsilon, lambda}, {w1});
    auto rsAg = pipeline.split(g1, AllReduceRSAG);
    auto reordered = pipeline.reorder({m1,v1,m_,v_,r,rNorm,wNorm,w1}, rsAg.second);
    pipeline.asSlice({m,v});
    pipeline.fuse(reordered.compStages);
    pipeline.print(std::cout);
    pipeline.codegen("lamb-rs-c-ag.cu");
}

// void lambRS_C_AG()
// {
//     Variable N(Int32, "N");
//     Variable lr(Float32, "lr");
//     Variable beta1(Float32, "beta1");
//     Variable beta2(Float32, "beta2");
//     Variable epsilon(Float32, "epsilon");
//     Variable lambda(Float32, "gamma");
//     Tensor w(Float32, N, Replicated, "w");
//     Tensor g(Float32, N, Local, "g");
//     Tensor m(Float32, N, Sliced, "m");
//     Tensor v(Float32, N, Sliced, "v");

//     Stage g1 = ReduceScatter(Summation, g);
//     Stage m1 = Update(m, beta1 * m + (1.0 - beta1) * g1);
//     Stage v1 = Update(v, beta2 * v + (1.0 - beta2) * g1 * g1);
//     Stage m_ = m1 / (1.0 - beta1);
//     Stage v_ = v1 / (1.0 - beta2);
//     Stage r = m_ / Sqrt(v_) + lambda * w;
//     Stage rNorm = Norm(r);
//     Stage wNorm = Norm(w);
//     Stage w1 = w - lr * (wNorm/rNorm) * r;
//     Stage w2 = Update(w, AllGather(w1));

//     Pipeline pipeline("lamb", {g, w, m, v, lr, beta1, beta2, epsilon, lambda}, {w2});
//     pipeline.fuse({m1, v1, m_, v_, r, rNorm, wNorm, w1});
//     // pipeline.print(std::cout);
//     pipeline.codegen("lamb-rs-c-ag.cu");
// }

void lamb_fuse_RS_C_AG()
{
    Variable N(Int32, "N");
    Variable lr(Float32, "lr");
    Variable beta1(Float32, "beta1");
    Variable beta2(Float32, "beta2");
    Variable epsilon(Float32, "epsilon");
    Variable lambda(Float32, "gamma");
    Tensor w(Float32, N, Replicated, "w");
    Tensor g(Float32, N, Local, "g");
    Tensor m(Float32, N, Sliced, "m");
    Tensor v(Float32, N, Sliced, "v");

    Stage g1 = ReduceScatter(Summation, g);
    Stage m1 = Update(m, beta1 * m + (1.0 - beta1) * g1);
    Stage v1 = Update(v, beta2 * v + (1.0 - beta2) * g1 * g1);
    Stage m_ = m1 / (1.0 - beta1);
    Stage v_ = v1 / (1.0 - beta2);
    Stage r = m_ / Sqrt(v_) + lambda * w;
    Stage rNorm = Norm(r);
    Stage wNorm = Norm(w);
    Stage w1 = w - lr * (wNorm/rNorm) * r;
    Stage w2 = Update(w, AllGather(w1));

    Pipeline pipeline("lamb", {g, w, m, v, lr, beta1, beta2, epsilon, lambda}, {w2});
    pipeline.fuse({g1, m1, v1, m_, v_, r, rNorm, wNorm, w1, w2});
    // pipeline.print(std::cout);
    pipeline.codegen("lamb-fuse-rs-c-ag.cu");
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./" << argv[0] << "<schedule>\n <schedule> is AR_C, RS_C_AG, fuse(RS_C_AG)\n" << std::endl;
        return 0;
    }
    
    std::string schedule = std::string(argv[1]);

    if (schedule == "AR_C")
        lambAR_C();
    else if (schedule == "RS_C_AG")
        lambRS_C_AG();
    else if (schedule == "fuse_RS_C_AG")
        lamb_fuse_RS_C_AG();
    else 
        std::cout << "Invalid schedule" << std::endl;

    return 0;    
}
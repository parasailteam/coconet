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

    Pipeline pipeline({g, w, m, v, lr, beta1, beta2, epsilon, lambda}, {w1});
    pipeline.fuse({m1,v1,m_,v_,r,rNorm,wNorm,w1});
    // pipeline.print(std::cout);
    pipeline.codegen("lamb-ar-c.cu");
}

void lambRS_C_AG()
{
    #if 0
    int N = 1024;
    int P = 4;

    Variable lr(Float32, P, "lr");
    Variable beta1(Float32, P, "beta1");
    Variable beta2(Float32, P, "beta2");
    Variable epsilon(Float32, P, "epsilon");
    Variable gamma(Float32, P, "gamma");
    Tensor w(Float32, N, P, "w");
    Tensor g(Float32, N, P, "g");
    ScatteredTensor m(Float32, N, P, "m");
    ScatteredTensor v(Float32, N, P, "v");

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
    #endif
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
    else if (schedule == "fuse(RS_C_AG)")
        ;
    else 
        std::cout << "Invalid schedule" << std::endl;

    return 0;    
}
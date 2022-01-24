#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

using namespace ACCCDSL;

void adamAR_C() 
{
    Variable N(TensorElemType::Int32, "N");
    Variable lr(TensorElemType::Float32, "lr");
    Variable beta1(TensorElemType::Float32, "beta1");
    Variable beta2(TensorElemType::Float32, "beta2");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");
    Tensor m(TensorElemType::Float32, N, Replicated, "m");
    Tensor v(TensorElemType::Float32, N, Replicated, "v");
    
    Stage g1 = AllReduce(Summation, g);
    Stage m1 = Update(m, beta1*m + (1-beta1)*g1);
    Stage v1 = Update(v, beta2*v + (1-beta2)*g1*g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = Update(w, w - m_/v_);

    Pipeline pipeline("adam", {g, w, m, v, lr, beta1, beta2}, {m1, v1, w1});
    pipeline.fuse({m1,v1,m_,v_,w1});
    pipeline.codegen("adam-ar-c.cu");
}

void adamRS_C_AG() 
{
    Variable N(TensorElemType::Int32, "N");
    Variable lr(TensorElemType::Float32, "lr");
    Variable beta1(TensorElemType::Float32, "beta1");
    Variable beta2(TensorElemType::Float32, "beta2");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");
    Tensor m(TensorElemType::Float32, N, Replicated, "m");
    Tensor v(TensorElemType::Float32, N, Replicated, "v");
    
    Stage g1 = AllReduce(Summation, g);
    Stage m1 = Update(m, beta1*m + (1-beta1)*g1);
    Stage v1 = Update(v, beta2*v + (1-beta2)*g1*g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = Update(w, w - m_/v_);

    Pipeline pipeline("adam", {g, w, m, v, lr, beta1, beta2}, {m1, v1, w1});
    auto rsAg = pipeline.split(g1, AllReduceRSAG);
    auto reordered = pipeline.reorder({m1,v1,m_,v_,w1}, rsAg.second);
    pipeline.asSlice({m,v});
    // pipeline.fuse(reordered.compStages);
    pipeline.print(std::cout);
    pipeline.codegen("adam-rs-c-ag.cu");
}

// void adamRS_C_AG()
// {
//     Variable N(TensorElemType::Int32, "N");
//     Variable lr(TensorElemType::Float32, "lr");
//     Variable beta1(TensorElemType::Float32, "beta1");
//     Variable beta2(TensorElemType::Float32, "beta2");
//     Tensor g(TensorElemType::Float32, N, Local, "g");
//     Tensor w(TensorElemType::Float32, N, Replicated, "w");
//     Tensor m(TensorElemType::Float32, N, Sliced, "m");
//     Tensor v(TensorElemType::Float32, N, Sliced, "v");
    
//     Stage g1 = ReduceScatter(Summation, g);
//     Stage m1 = Update(m, beta1*m + (1-beta1)*g1);
//     Stage v1 = Update(v, beta2*v + (1-beta2)*g1*g1);
//     Stage m_ = m1/beta1;
//     Stage v_ = v1/beta2;
//     Stage w1 = Scatter(w) - m_/v_;
//     Stage w2 = Update(w, AllGather(w1));

//     Pipeline pipeline("adam", {g, w, m, v, lr, beta1, beta2}, {m1, v1, w2});
//     pipeline.fuse({m1,v1,m_,v_,w1});
//     pipeline.codegen("adam-rs-c-ag.cu");
// }

void adamfuseRS_C_AG()
{
    Variable N(TensorElemType::Int32, "N");
    Variable lr(TensorElemType::Float32, "lr");
    Variable beta1(TensorElemType::Float32, "beta1");
    Variable beta2(TensorElemType::Float32, "beta2");
    Tensor g(TensorElemType::Float32, N, Local, "g");
    Tensor w(TensorElemType::Float32, N, Replicated, "w");
    Tensor m(TensorElemType::Float32, N, Sliced, "m");
    Tensor v(TensorElemType::Float32, N, Sliced, "v");
    
    Stage g1 = ReduceScatter(Summation, g);
    Stage m1 = Update(m, beta1*m + (1-beta1)*g1);
    Stage v1 = Update(v, beta2*v + (1-beta2)*g1*g1);
    Stage m_ = m1/beta1;
    Stage v_ = v1/beta2;
    Stage w1 = Scatter(w) - m_/v_;
    Stage w2 = Update(w, AllGather(w1));

    Pipeline pipeline("adam", {g, w, m, v, lr, beta1, beta2}, {m1, v1, w2});
    pipeline.fuse({g1,m1,v1,m_,v_,w1,w2});
    pipeline.codegen("adam-fuse-rs-c-ag.cu");
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./" << argv[0] << "<schedule>\n <schedule> is AR_C, RS_C_AG, fuse(RS_C_AG)\n" << std::endl;
        return 0;
    }
    
    std::string schedule = std::string(argv[1]);

    if (schedule == "AR_C")
        adamAR_C();
    else if (schedule == "RS_C_AG")
        adamRS_C_AG();
    else if (schedule == "fuse_RS_C_AG")
        adamfuseRS_C_AG();
    else 
        std::cout << "Invalid schedule" << std::endl;

    return 0;
}
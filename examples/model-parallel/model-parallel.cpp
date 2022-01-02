#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

using namespace ACCCDSL;

void MM_AR_C() 
{
    Variable B(Int32, "B");
    Variable S(Int32, "S");    
    Variable H(Int32, "H");

    Tensor w(Float16, {H,H}, Sliced, "w");
    Tensor b(Float16, H, Replicated, "b");
    Tensor in(Float16, {B,S,H}, Sliced_2, "in");
    Tensor r(Float16, {B,S,H}, Replicated, "r");

    Stage layer = MatMul(in,w);
    Stage sum = AllReduce(Summation, layer);
    Stage out = Dropout(sum, 0.5);
    
    Pipeline pipeline("model-parallel", {w,b,in,r}, {out});

    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("model-parallel-mm-ar-c.cu", varBounds);
}

void MM_RS_C_AG()
{
    Variable B(Int32, "B");
    Variable S(Int32, "S");    
    Variable H(Int32, "H");

    Tensor w(Float16, {H,H}, Sliced, "w");
    Tensor b(Float16, H, Replicated, "b");
    Tensor in(Float16, {B,S,H}, Sliced_2, "in");
    Tensor r(Float16, {B,S,H}, Replicated, "r");

    Stage layer = MatMul(in,w);
    Stage sumRS = ReduceScatter(Summation, layer);
    Stage scOut = sumRS + Scatter(r);
    Stage out = AllGather(scOut);

    Pipeline pipeline("model-parallel", {w,b,in,r}, {out});
    
    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("model-parallel-mm-rs-c-ag.cu", varBounds);
}

void ol_MM_fuse_RS_C_AG()
{
    Variable B(Int32, "B");
    Variable S(Int32, "S");    
    Variable H(Int32, "H");

    Tensor w(Float16, {H,H}, Sliced, "w");
    Tensor b(Float16, H, Replicated, "b");
    Tensor in(Float16, {B,S,H}, Sliced_2, "in");
    Tensor r(Float16, {B,S,H}, Replicated, "r");

    Stage layer = MatMul(in,w);
    Stage sumRS = ReduceScatter(Summation, layer);
    Stage scOut = sumRS + Scatter(r);
    Stage out = AllGather(scOut);

    Pipeline pipeline("model-parallel", {w,b,in,r}, {out});

    pipeline.fuse({sumRS, scOut, out});
    pipeline.overlap({layer, sumRS, scOut, out});

    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("model-parallel-ol-mm-fuse-rs-c-ag.cu", varBounds);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./" << argv[0] << "<schedule>\n <schedule> is AR_C, RS_C_AG, fuse(RS_C_AG)\n" << std::endl;
        return 0;
    }
    
    std::string schedule = std::string(argv[1]);

    if (schedule == "MM_AR_C")
        MM_AR_C();
    else if (schedule == "MM_RS_C_AG")
        MM_RS_C_AG();
    else if (schedule == "ol_MM_fuse_RS_C_AG")
        ol_MM_fuse_RS_C_AG();
    else 
        std::cout << "Invalid schedule '" << schedule << "'" << std::endl;

    return 0;
}
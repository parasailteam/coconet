#include <pipeline.hpp>
#include <dsl.hpp>
#include <astvisitor.hpp>

using namespace ACCCDSL;

void AR_P2P_C() 
{
    Variable B(Int32, "B");
    Variable S(Int32, "S");    
    Variable H(Int32, "H");

    Tensor w(Float16, {H,H}, Sliced, "w");
    Tensor b(Float16, H, Replicated, "b");
    Tensor in(Float16, {B,S,H}, Local, "in");
    Tensor r(Float16, {B,S,H}, Replicated, "r");
    
    Stage sum = AllReduce(Summation, in);
    Stage recv = Send(sum, NextGroupRank(GROUP, RANK));
    Stage out = Dropout(recv+b, 0.1) + r;

    Pipeline transformer({in}, {out});
    
    Pipeline pipeline("pipeline-parallel", {w,b,in,r}, {out});

    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("pipeline-parallel-ar-p2p-c.cu", varBounds);
}

void RS_P2P_C_AG()
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

    Pipeline pipeline("pipeline-parallel", {w,b,in,r}, {out});
    
    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("pipeline-parallel-rs-p2p-c-ag.cu", varBounds);
}

void ol_RS_fuse_P2P_C_AG()
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

    Pipeline pipeline("pipeline-parallel", {w,b,in,r}, {out});

    pipeline.fuse({sumRS, scOut, out});
    pipeline.overlap({layer, sumRS, scOut, out});

    std::vector<CodeGenVarBounds> varBounds = {CodeGenVarBounds(B, {8, 16}), CodeGenVarBounds(S, {1024}), CodeGenVarBounds(H, {3072})};
    pipeline.codegen("pipeline-parallel-ol-rs-fuse-p2p-c-ag.cu", varBounds);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./" << argv[0] << "<schedule>\n <schedule> is AR_C, RS_C_AG, fuse(RS_C_AG)\n" << std::endl;
        return 0;
    }
    
    std::string schedule = std::string(argv[1]);

    if (schedule == "AR_P2P_C")
        AR_P2P_C();
    else if (schedule == "RS_P2P_C_AG")
        RS_P2P_C_AG();
    else if (schedule == "ol_RS_fuse_P2P_C_AG")
        ol_RS_fuse_P2P_C_AG();
    else 
        std::cout << "Invalid schedule '" << schedule << "'" << std::endl;

    return 0;
}
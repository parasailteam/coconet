#include "utils.hpp"
#include "dsl.hpp"

size_t sizeOfElemType(ACCCDSL::TensorElemType t)
{
    switch (t) {
        case ACCCDSL::TensorElemType::Float16:
            return 2;
        case ACCCDSL::TensorElemType::Float32:
            return 4;
        case ACCCDSL::TensorElemType::Float64:
            return 8;
        case ACCCDSL::TensorElemType::Int8:
            return 1;
        case ACCCDSL::TensorElemType::Int16:
            return 2;
        case ACCCDSL::TensorElemType::Int32:
            return 4;
        case ACCCDSL::TensorElemType::Int64:
            return 8;
        default:
            ASSERT(false, "Unimplemented type");
            return 0;
    }
}

std::string elemTypeToCType(ACCCDSL::TensorElemType t)
{
    switch (t) {
        case ACCCDSL::TensorElemType::Float16:
            return "half";
        case ACCCDSL::TensorElemType::Float32:
            return "float";
        case ACCCDSL::TensorElemType::Float64:
            return "double";
        case ACCCDSL::TensorElemType::Int8:
            return "char";
        case ACCCDSL::TensorElemType::Int16:
            return "short int";
        case ACCCDSL::TensorElemType::Int32:
            return "int";
        case ACCCDSL::TensorElemType::Int64:
            return "long";
        default:
            ASSERT(false, "Unimplemented type");
            return "";
    }
}

std::string elemTypeToNCCLType(ACCCDSL::TensorElemType t)
{
    switch (t) {
        case ACCCDSL::TensorElemType::Float16:
            return "ncclHalf";
        case ACCCDSL::TensorElemType::Float32:
            return "ncclFloat32";
        case ACCCDSL::TensorElemType::Float64:
            return "ncclFloat64";
        case ACCCDSL::TensorElemType::Int8:
            return "ncclInt8";
        case ACCCDSL::TensorElemType::Int16:
            return "ncclInt16";
        case ACCCDSL::TensorElemType::Int32:
            return "ncclInt32";
        case ACCCDSL::TensorElemType::Int64:
            return "ncclInt64";
        default:
            ASSERT(false, "Unimplemented type");
            return "";
    }
}

std::string elemTypeToMPIType(ACCCDSL::TensorElemType t)
{
    switch (t) {
        case ACCCDSL::TensorElemType::Float16:
            return "MPI_FLOAT";
        case ACCCDSL::TensorElemType::Float32:
            return "MPI_FLOAT";
        case ACCCDSL::TensorElemType::Float64:
            return "MPI_DOUBLE";
        case ACCCDSL::TensorElemType::Int8:
            return "MPI_CHAR";
        case ACCCDSL::TensorElemType::Int16:
            return "MPI_SHORT";
        case ACCCDSL::TensorElemType::Int32:
            return "MPI_INT";
        case ACCCDSL::TensorElemType::Int64:
            return "MPI_LONG";
        default:
            ASSERT(false, "Unimplemented type");
            return "";
    }
}

std::string redOpToNCCLReduceOp(ACCCDSL::ReduceOperation op) 
{
    switch (op) {
        case ACCCDSL::Summation:
            return "ncclSum";
        case ACCCDSL::Difference:
            return "ncclDiff"; //TODO: Not supported by nccl.
        case ACCCDSL::Multiplication:
            return "ncclProf";
        case ACCCDSL::Division:
            return "ncclDiv"; //TODO: Not supported by nccl.
        case ACCCDSL::Maximum:
            return "ncclMax";
        case ACCCDSL::Minimum:
            return "ncclMin";
        default:
            ASSERT(false, "Unimplemented reduction type");
            return "";
    }
}

std::string redOpToMPIReduceOp(ACCCDSL::ReduceOperation op) 
{
    switch (op) {
        case ACCCDSL::Summation:
            return "MPI_SUM";
        case ACCCDSL::Difference:
            return "MPI-DIV"; //TODO: Not supported by MPI.
        case ACCCDSL::Multiplication:
            return "MPI_PROD";
        case ACCCDSL::Division:
            return "MPI_DIV"; //TODO: Not supported by MPI.
        case ACCCDSL::Maximum:
            return "MPI_MAX";
        case ACCCDSL::Minimum:
            return "MPI_MIN";
        default:
            ASSERT(false, "Unimplemented reduction type");
            return "";
    }
}

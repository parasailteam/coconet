#include "utils.hpp"
#include "dsl.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string.h>
#include <unistd.h>
#include <regex>

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

std::string elemTypeToCUBLASType(ACCCDSL::TensorElemType t)
{
    switch (t) {
        case ACCCDSL::TensorElemType::Float16:
            return "CUDA_R_16F";
        case ACCCDSL::TensorElemType::Float32:
            return "CUDA_R_32F";
        case ACCCDSL::TensorElemType::Float64:
            return "CUDA_R_64F";
        case ACCCDSL::TensorElemType::Int8:
            return "CUDA_C_8I";
        case ACCCDSL::TensorElemType::Int16:
            return "CUDA_C_16I";
        case ACCCDSL::TensorElemType::Int32:
            return "CUDA_C_32I";
        case ACCCDSL::TensorElemType::Int64:
            return "CUDA_C_64I";
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

std::string replaceAllSubString(std::string& s, std::string subs, std::string replacement)
{
    while(s.find(subs) != s.npos) {
        s = s.replace(s.find(subs), subs.size(), replacement);
    }

    return s;
}

std::string readFile(std::string filepath) 
{
    std::ifstream ifs(filepath.c_str(), std::ifstream::in);
    if (!ifs.is_open()) {
        ASSERT(false, "Cannot open file '"<<filepath <<"'");
    }
    std::ostringstream sstr;
    sstr << ifs.rdbuf();
    ifs.close();
    return sstr.str();
}

void writeFile(std::string filepath, const std::string& contents) 
{
    std::ofstream file(filepath.c_str(), std::ofstream::out);
    if (!file.is_open()) {
        ASSERT(false, "Cannot open file '" << filepath << "'");
    }

    file.write(contents.c_str(), contents.size());
    file.close();
}

void writeFile(int fd, const std::string& contents) 
{
    write(fd, contents.c_str(), contents.size());
    close(fd);
}

std::string exec(const std::string& cmd) 
{
    std::array<char, 128> buffer;
    std::string result;

    auto pipe = popen(cmd.c_str(), "r"); // get rid of shared_ptr

    if (!pipe) throw std::runtime_error("popen() failed!");

    while (!feof(pipe)) {
        if (fgets(buffer.data(), 128, pipe) != nullptr)
            result += buffer.data();
    }

    auto rc = pclose(pipe);

    if (rc == EXIT_SUCCESS) { // == 0

    } else {  // EXIT_FAILURE is not used by all programs, maybe needs some adaptation.
        std::cout << "executing '" << cmd << "' failed with " << rc << std::endl;
        std::cout << "output " << result << std::endl;
        ASSERT(false, "");
    }
    return result;
}

uint32_t nextPowerOf2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

uint64_t nextPowerOf2(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;

    return v;
}

uint64_t isPowerOf2(uint64_t num) {
    return ((num != 0) && ((num &(num - 1)) == 0));
}

uint64_t currOrNextPowerOf2(uint64_t num) {
    if (isPowerOf2(num))
        return num;
    return nextPowerOf2(num);
}

void replaceAllSubStringInFile(std::string filepath, std::string regexSub, std::string replacement)
{
    std::regex e(regexSub);
    std::string contents = readFile(filepath);
    contents = std::regex_replace(contents, e, replacement);
    writeFile(filepath, contents);
}

std::string indent(int level) 
{
    std::string s = "";
    for (int i = 0; i < level; i++) {
        s += "  ";
    }
    return s;
}
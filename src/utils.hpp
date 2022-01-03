#include<string>

#include "ast.hpp"

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

std::string elemTypeToCType(ACCCDSL::TensorElemType t);
std::string elemTypeToNCCLType(ACCCDSL::TensorElemType t);
std::string elemTypeToMPIType(ACCCDSL::TensorElemType t);
std::string redOpToNCCLReduceOp(ACCCDSL::ReduceOperation op);
std::string redOpToMPIReduceOp(ACCCDSL::ReduceOperation op);
size_t sizeOfElemType(ACCCDSL::TensorElemType t);
std::string elemTypeToCUBLASType(ACCCDSL::TensorElemType t);
/**File and string functions**/
std::string replaceAllSubString(std::string& s, std::string subs, std::string replacement);
std::string readFile(std::string filepath);
void writeFile(std::string filepath, const std::string& contents);
void writeFile(int fd, const std::string& contents);
void replaceAllSubStringInFile(std::string filepath, std::string regexSub, std::string replacement);
std::string indent(int level);

std::string exec(const std::string& cmd);
/*Power of two functions*/
uint32_t nextPowerOf2(uint32_t v);
uint64_t nextPowerOf2(uint64_t v);
uint64_t isPowerOf2(uint64_t num);
uint64_t currOrNextPowerOf2(uint64_t num);
#endif /*__UTILS_HPP__*/
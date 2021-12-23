#ifndef __CODEGEN_HPP__
#define __CODEGEN_HPP__

#include "pipeline.hpp"

#include <iostream>

namespace ACCCDSLImpl {
    
    class CodeGenVarBounds {
        protected:
        Variable var_;
        CodeGenVarBounds(Variable var) : var_(var) {}
        virtual ~CodeGenVarBounds();
    };
    
    class CodeGenVarRange : public CodeGenVarBounds {
        int start_;
        int end_;
        int increment_;

        public:
        CodeGenVarRange(Variable var, int start, int end, int increment) :
            CodeGenVarBounds(var), 
            start_(start), end_(end), increment_(increment) {}
    };

    class CodeGenVarLogRange : public CodeGenVarBounds {
        int start_;
        int end_;
        int increment_;

        public:
        CodeGenVarLogRange(Variable var, int start, int end, int increment) :
            CodeGenVarBounds(var), 
            start_(start), end_(end), increment_(increment) {}
    };

    class CodeGenVarValues : public CodeGenVarBounds {
        std::vector<int> values_;

        public:
        CodeGenVarValues(Variable var, std::vector<int> values) :
            CodeGenVarBounds(var), values_(values) {}
    };

    class Codegen
    {
        protected:
        std::ostream& os_;
        Pipeline& pipeline_;
        int options_;

        public:
        Codegen(Pipeline& pipeline, std::ostream& os, int options) : os_(os), pipeline_(pipeline), options_(options) {}
        virtual void codegen() = 0;
    };
    
    class NCCLCodegen : public Codegen
    {
        public:
        NCCLCodegen(Pipeline& pipeline, std::ostream& os) : Codegen(pipeline, os, (int)(CodeGenOptions::GenMultiProcessCode | CodeGenOptions::GenMainFunction)) {}
        virtual void codegen();
    };
}

#endif
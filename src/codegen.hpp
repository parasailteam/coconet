#ifndef __CODEGEN_HPP__
#define __CODEGEN_HPP__

#include "pipeline.hpp"

#include <iostream>

namespace ACCCDSLImpl {
    class Codegen
    {
        protected:
        std::ostream& os_;
        Pipeline& pipeline_;
        int options_;

        public:
        Codegen(Pipeline& pipeline, std::ostream& os, int options) : os_(os), pipeline_(pipeline), options_(options) {}
        virtual void codegen(std::vector<CodeGenVarBounds> varBounds) = 0;
    };
    
    class NCCLCodegen : public Codegen
    {
        public:
        NCCLCodegen(Pipeline& pipeline, std::ostream& os) : Codegen(pipeline, os, (int)(CodeGenOptions::GenMultiProcessCode | CodeGenOptions::GenMainFunction)) {}
        virtual void codegen(std::vector<CodeGenVarBounds> varBounds);
    };
}

#endif
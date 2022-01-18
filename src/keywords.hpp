#include "dsl.hpp"

#ifndef __KEYWORDS_HPP__
#define __KEYWORDS_HPP__

namespace ACCCDSL {
extern Variable RANK;
extern ProcessGroup WORLD;

SingleDimExpression NextGroupRank(SingleDimExpression group, SingleDimExpression rank);
}

#endif
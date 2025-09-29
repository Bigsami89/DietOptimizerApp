#pragma once
#include "IOptimizer.h"
#include "ApiRequest.h"

class RationCalculator {
public:
    explicit RationCalculator(std::unique_ptr<IOptimizer> solver);
    std::vector<Diet> findTopDiets(const ApiRequest& request);
private:
    std::unique_ptr<IOptimizer> optimizer;
};
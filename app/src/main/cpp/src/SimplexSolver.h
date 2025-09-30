#pragma once
#include "IOptimizer.h"
class SimplexSolver : public IOptimizer {
public:
    std::optional<Diet> solve(
            const std::vector<Ingredient>& ingredients,
            const std::map<std::string, double>& minNutrients,
            const std::map<std::string, double>& maxNutrients,
            double costWeight,
            double methaneWeight,
            double DMI_kg_day,
            double body_weight_kg = 0.0 // NUEVO
    ) override ;
};
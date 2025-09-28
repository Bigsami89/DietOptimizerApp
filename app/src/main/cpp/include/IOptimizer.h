#pragma once
#include "Diet.h"
#include <vector>
#include <optional>
#include <map>

class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual std::optional<Diet> solve(
        const std::vector<Ingredient>& ingredients,
        const std::map<std::string, double>& minNutrients,
        const std::map<std::string, double>& maxNutrients,
        double costWeight,
        double methaneWeight,
        double DMI_kg_day
    ) = 0;
};
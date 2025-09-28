#pragma once
#include "Ingredient.h"
#include <string>
#include <vector>
#include <map>

struct ApiRequest {
    std::map<std::string, double> minRequirements;
    std::map<std::string, double> maxRequirements;
    std::vector<Ingredient> availableIngredients;
    double DMI_kg_day = 0.0;

    static bool fromJson(const std::string& jsonString, ApiRequest& request);
};
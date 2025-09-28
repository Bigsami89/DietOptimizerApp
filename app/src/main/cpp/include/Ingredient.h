#pragma once
#include <string>
#include <map>

class Ingredient {
public:
    std::string name;
    double cost;
    std::map<std::string, double> nutrients;
    bool isForage = false;
    

    // DECLARACIÓN del constructor simplificado (sin min/max inclusion)
    Ingredient(std::string name, double cost, std::map<std::string, double> nutrients);

    double getNutrient(const std::string& key) const;
};
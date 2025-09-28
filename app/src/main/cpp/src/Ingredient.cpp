#include "Ingredient.h"

// DEFINICIÓN del constructor simplificado. Ahora coincide con la declaración en el .h
Ingredient::Ingredient(std::string name, double cost, std::map<std::string, double> nutrients)
    : name(std::move(name)), cost(cost), nutrients(std::move(nutrients)) {}

double Ingredient::getNutrient(const std::string& key) const {
    auto it = nutrients.find(key);
    return (it != nutrients.end()) ? it->second : 0.0;
}
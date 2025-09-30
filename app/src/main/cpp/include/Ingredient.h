#pragma once
#include <string>
#include <map>

class Ingredient {
public:
    std::string name;
    double cost;
    std::map<std::string, double> nutrients;
    bool isForage = false; // Nuevo: indica si el ingrediente es forraje

    // Constructor simplificado
    Ingredient(std::string name, double cost, std::map<std::string, double> nutrients, bool isForage = false);

    double getNutrient(const std::string& key) const;

    /**
     * @brief Determina automáticamente si un ingrediente es forraje basado en su composición
     * Criterio: NDF > 25% y CP < 20% típicamente indica forraje
     */
    void autoDetectForage();
};
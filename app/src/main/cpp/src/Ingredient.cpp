#include "Ingredient.h"

// Constructor actualizado con parámetro isForage
Ingredient::Ingredient(std::string name, double cost, std::map<std::string, double> nutrients, bool isForage)
        : name(std::move(name)), cost(cost), nutrients(std::move(nutrients)), isForage(isForage) {

    // Si no se especificó, intentar auto-detectar
    if (!isForage) {
        autoDetectForage();
    }
}

double Ingredient::getNutrient(const std::string& key) const {
    auto it = nutrients.find(key);
    return (it != nutrients.end()) ? it->second : 0.0;
}

void Ingredient::autoDetectForage() {
    // Criterios para clasificar como forraje según NASEM 2016:
    // 1. Alto contenido de fibra (NDF > 25%)
    // 2. Bajo a moderado contenido de proteína (CP < 20% típicamente)
    // 3. Moderado contenido energético (TDN < 75% típicamente)

    double ndf = getNutrient("NDF");
    double cp = getNutrient("CP");
    double tdn = getNutrient("TDN");

    // Clasificación automática
    if (ndf > 25.0 && cp < 20.0) {
        isForage = true;
    } else if (ndf > 35.0) {
        // Alto NDF casi siempre indica forraje
        isForage = true;
    } else if (tdn < 65.0 && ndf > 20.0) {
        // Baja digestibilidad con fibra moderada
        isForage = true;
    } else {
        isForage = false;
    }
}
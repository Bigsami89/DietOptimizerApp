#include "Diet.h"
void Diet::calculateDietProperties() {
    totalCost = 0.0;
    finalNutrientProfile.clear();
    for (const auto& pair : composition) {
        totalCost += pair.first.cost * pair.second;
        for (const auto& nutrient : pair.first.nutrients) {
            finalNutrientProfile[nutrient.first] += nutrient.second * pair.second;
        }
    }
}
void Diet::calculateEntericMethane() {
    double totalTDN = finalNutrientProfile["TDN"];
    entericMethane = 2.5 * totalTDN;
}
void Diet::print() const {
    std::cout << "-------------------------------------------\n";
    std::cout << "Dieta | Costo: " << totalCost << " | Metano: " << entericMethane << "\n";
    for (const auto& pair : composition) {
        std::cout << " - " << pair.first.name << ": " << (pair.second * 100.0) << "%\n";
    }
    std::cout << "-------------------------------------------\n\n";
}
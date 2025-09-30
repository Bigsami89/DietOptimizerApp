#include "RationCalculator.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

RationCalculator::RationCalculator(std::unique_ptr<IOptimizer> solver)
        : optimizer(std::move(solver)) {}

bool areDietsEqual(const Diet& a, const Diet& b) {
    if (a.composition.size() != b.composition.size()) {
        return false;
    }

    for (const auto& compA : a.composition) {
        bool foundMatch = false;
        for (const auto& compB : b.composition) {
            if (compA.first.name == compB.first.name && std::abs(compA.second - compB.second) < 0.001) {
                foundMatch = true;
                break;
            }
        }
        if (!foundMatch) {
            return false;
        }
    }
    return true;
}

std::vector<Diet> RationCalculator::findTopDiets(const ApiRequest& request) {

    std::vector<Diet> allResults;
    int numberOfSteps = 11;

    std::cout << "\nBuscando " << numberOfSteps -1 << " dietas optimas en el balance Costo-Metano...\n";
    std::cout << "Peso corporal del animal: " << request.body_weight_kg << " kg\n"; // Información de debug

    for (int i = 0; i < numberOfSteps; ++i) {
        double weight = static_cast<double>(i) / (numberOfSteps - 1);

        double costWeight = 1.0 - weight;
        double methaneWeight = weight;

        std::cout << "\n--- Calculando Dieta " << i + 1 << "/" << numberOfSteps
                  << " (Peso Costo: " << costWeight * 100 << "%, Peso Metano: " << methaneWeight * 100 << "%) ---\n";

        // Pasar el peso corporal al optimizador
        auto dietResult = optimizer->solve(
                request.availableIngredients,
                request.minRequirements,
                request.maxRequirements,
                costWeight,
                methaneWeight,
                request.DMI_kg_day,
                request.body_weight_kg // NUEVO: Peso corporal
        );

        if (dietResult.has_value()) {
            allResults.push_back(dietResult.value());
        } else {
            std::cerr << "No se pudo encontrar una solucion factible para esta ponderacion.\n";
        }
    }

    // Filtrado de duplicados
    std::vector<Diet> uniqueResults;
    if (!allResults.empty()) {
        std::sort(allResults.begin(), allResults.end(), [](const Diet& a, const Diet& b) {
            return a.totalCost < b.totalCost;
        });

        uniqueResults.push_back(allResults.front());

        for (const auto& currentDiet : allResults) {
            bool isDuplicate = false;
            for (const auto& uniqueDiet : uniqueResults) {
                if (areDietsEqual(currentDiet, uniqueDiet)) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) {
                uniqueResults.push_back(currentDiet);
            }
        }
    }

    return uniqueResults;
}
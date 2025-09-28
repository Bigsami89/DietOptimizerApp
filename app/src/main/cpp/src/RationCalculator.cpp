#include "RationCalculator.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath> // Necesario para std::abs

/**
 * @brief Constructor de la clase RationCalculator.
 * @param solver Un unique_ptr al optimizador que se utilizará para resolver las dietas.
 * Esto es un ejemplo de Inyección de Dependencias, lo que permite cambiar
 * el algoritmo de optimización fácilmente.
 */
RationCalculator::RationCalculator(std::unique_ptr<IOptimizer> solver)
    : optimizer(std::move(solver)) {}

/**
 * @brief Función auxiliar para determinar si dos dietas son funcionalmente idénticas.
 * Compara los ingredientes y sus proporciones con una pequeña tolerancia para
 * evitar problemas de precisión con números de punto flotante.
 * @param a La primera dieta a comparar.
 * @param b La segunda dieta a comparar.
 * @return true si las dietas son consideradas iguales, false en caso contrario.
 */
bool areDietsEqual(const Diet& a, const Diet& b) {
    // Si el número de ingredientes es diferente, no pueden ser iguales.
    if (a.composition.size() != b.composition.size()) {
        return false;
    }

    // Compara cada componente de la dieta 'a' con los de la dieta 'b'
    for (const auto& compA : a.composition) {
        bool foundMatch = false;
        for (const auto& compB : b.composition) {
            // Un componente coincide si el nombre del ingrediente es el mismo y
            // la proporción es casi idéntica (diferencia menor a 0.1%).
            if (compA.first.name == compB.first.name && std::abs(compA.second - compB.second) < 0.001) {
                foundMatch = true;
                break;
            }
        }
        // Si no se encontró una coincidencia para un componente de 'a', las dietas son diferentes.
        if (!foundMatch) {
            return false;
        }
    }
    // Si todos los componentes de 'a' tienen una coincidencia en 'b', las dietas son iguales.
    return true;
}

/**
 * @brief Encuentra las mejores dietas únicas explorando el balance entre costo y metano.
 * Este método genera un "frente de Pareto", mostrando las soluciones óptimas
 * donde no se puede mejorar un objetivo (ej. costo) sin empeorar el otro (ej. metano).
 * @param request La solicitud completa del API, conteniendo los requerimientos y los ingredientes.
 * @return Un vector de objetos Diet, donde cada una es una solución óptima y única.
 */
std::vector<Diet> RationCalculator::findTopDiets(const ApiRequest& request) {
    
    std::vector<Diet> allResults;
    int numberOfSteps = 11; // Genera 11 puntos de prueba (para w = 0.0, 0.1, ..., 1.0)

    std::cout << "\nBuscando " << numberOfSteps -1 << " dietas optimas en el balance Costo-Metano...\n";

    // Bucle principal: se ejecuta una vez por cada ponderación
    for (int i = 0; i < numberOfSteps; ++i) {
        double weight = static_cast<double>(i) / (numberOfSteps - 1);
        
        double costWeight = 1.0 - weight; // El peso del costo disminuye
        double methaneWeight = weight;      // El peso del metano aumenta

        std::cout << "\n--- Calculando Dieta " << i + 1 << "/" << numberOfSteps 
                  << " (Peso Costo: " << costWeight * 100 << "%, Peso Metano: " << methaneWeight * 100 << "%) ---\n";

        // Llama al optimizador con la ponderación actual
        auto dietResult = optimizer->solve(
            request.availableIngredients,
            request.minRequirements,
            request.maxRequirements,
            costWeight,
            methaneWeight,
            request.DMI_kg_day
        );

        // Si el optimizador encontró una solución válida, la guarda
        if (dietResult.has_value()) {
            allResults.push_back(dietResult.value());
        } else {
            std::cerr << "No se pudo encontrar una solucion factible para esta ponderacion.\n";
        }
    }
    
    // --- FILTRADO DE RESULTADOS DUPLICADOS ---
    std::vector<Diet> uniqueResults;
    if (!allResults.empty()) {
        // Ordena los resultados por costo para una presentación más lógica
        std::sort(allResults.begin(), allResults.end(), [](const Diet& a, const Diet& b) {
            return a.totalCost < b.totalCost;
        });

        // Agrega la primera dieta a la lista de resultados únicos para empezar
        uniqueResults.push_back(allResults.front());

        // Recorre el resto de las dietas generadas
        for (const auto& currentDiet : allResults) {
            bool isDuplicate = false;
            // Compara la dieta actual con cada una de las que ya están en la lista de únicos
            for (const auto& uniqueDiet : uniqueResults) {
                if (areDietsEqual(currentDiet, uniqueDiet)) {
                    isDuplicate = true;
                    break;
                }
            }
            // Si no se encontró ninguna coincidencia, es una dieta nueva y se agrega
            if (!isDuplicate) {
                uniqueResults.push_back(currentDiet);
            }
        }
    }
    
    return uniqueResults;
}
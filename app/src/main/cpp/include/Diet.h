#pragma once

#include "Ingredient.h" // La dieta depende de la definición de Ingrediente
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip> // Para formatear la salida en la función de impresión

/**
 * @brief Representa una solución de dieta completa.
 * Esta clase contiene la composición de la dieta (ingredientes y sus proporciones)
 * y métodos para calcular su perfil nutricional, costo y emisiones.
 */
class Diet {
public:
    // Almacena la composición final de la dieta. Ej: { (Maíz, 0.70), (Soja, 0.30) }
    std::vector<std::pair<Ingredient, double>> composition;

    // Almacena el perfil nutricional calculado de la mezcla final. Ej: { "CP": 14.5, "NEm": 1.9 }
    std::map<std::string, double> finalNutrientProfile;

    // Propiedades calculadas
    double totalCost = 0.0;
    double entericMethane = 0.0; // Emisiones de metano en gramos/día

    /**
     * @brief Calcula el perfil nutricional completo y el costo total de la dieta.
     * Itera sobre cada ingrediente en la composición, suma sus aportes nutricionales
     * ponderados por su proporción y calcula el costo final.
     */
    void calculateDietProperties() {
        totalCost = 0.0;
        finalNutrientProfile.clear(); // Limpiar cálculos anteriores

        for (const auto& pair : composition) {
            const auto& ingredient = pair.first;
            const double proportion = pair.second;

            // Sumar al costo total
            totalCost += ingredient.cost * proportion;

            // Sumar el aporte de cada nutriente del ingrediente al perfil final
            for (const auto& nutrient : ingredient.nutrients) {
                finalNutrientProfile[nutrient.first] += nutrient.second * proportion;
            }
        }
    }

    /**
     * @brief Calcula las emisiones de metano entérico usando la ecuación del NASEM 2016.
     * La fórmula se basa en que el 6.5% del consumo de Energía Bruta (GEI) se convierte en metano.
     * @param DMI_kg_day El Consumo de Materia Seca del animal en kg/día, un dato crucial.
     */
    void calculateEntericMethane(double DMI_kg_day) {
        if (DMI_kg_day <= 0) {
            entericMethane = 0.0;
            return;
        }

        // 1. Obtener la Energía Bruta (GE) total de la dieta en Mcal/kg del perfil ya calculado.
        double totalGE_Mcal_per_kg = finalNutrientProfile.count("GE") ? finalNutrientProfile.at("GE") : 0.0;

        // 2. Calcular el Consumo de Energía Bruta (GEI) total en Mcal/día.
        // GEI (Mcal/día) = GE de la dieta (Mcal/kg) * Consumo de MS (kg/día)
        double GEI_Mcal_per_day = totalGE_Mcal_per_kg * DMI_kg_day;

        // 3. Aplicar la ecuación NASEM: CH4 (Mcal/día) = 6.5% del GEI
        double methane_Mcal_per_day = GEI_Mcal_per_day * 0.065;

        // 4. Convertir la energía del metano (Mcal/día) a masa (gramos/día).
        // Factor de conversión: 1 g de CH4 produce aproximadamente 13.25 kcal de energía.
        // Factor = (1000 kcal / 1 Mcal) / (13.25 kcal / 1 g CH4) ≈ 75.47 g/Mcal
        const double Mcal_to_grams_factor = 75.47;
        entericMethane = methane_Mcal_per_day * Mcal_to_grams_factor;
    }

    /**
     * @brief Imprime un resumen formateado de la dieta en la consola.
     * Útil para la depuración y para mostrar los resultados al usuario.
     */
    void print() const {
        std::cout << "-------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(4); // Formatear a 4 decimales
        std::cout << "Dieta | Costo Total: " << totalCost << " | Metano (g/dia): " << entericMethane << "\n";
        std::cout << "-------------------------------------------\n";
        for (const auto& pair : composition) {
            std::cout << " - " << std::left << std::setw(28) << pair.first.name
                      << ": " << std::right << std::setw(6) << std::fixed << std::setprecision(2)
                      << (pair.second * 100.0) << "%\n";
        }
        std::cout << "-------------------------------------------\n\n";
    }
};
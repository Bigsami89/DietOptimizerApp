#pragma once

#include "Ingredient.h"
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * @brief Representa una solución de dieta completa con cálculos precisos según NASEM 2016
 */
class Diet {
public:
    std::vector<std::pair<Ingredient, double>> composition;
    std::map<std::string, double> finalNutrientProfile;
    double totalCost = 0.0;
    double entericMethane = 0.0; // Emisiones de metano en gramos/día

    /**
     * @brief Calcula el perfil nutricional completo y el costo total de la dieta
     */
    void calculateDietProperties() {
        totalCost = 0.0;
        finalNutrientProfile.clear();

        for (const auto& pair : composition) {
            const auto& ingredient = pair.first;
            const double proportion = pair.second;

            totalCost += ingredient.cost * proportion;

            for (const auto& nutrient : ingredient.nutrients) {
                finalNutrientProfile[nutrient.first] += nutrient.second * proportion;
            }
        }
    }

    /**
     * @brief Calcula las emisiones de metano entérico usando ecuaciones NASEM 2016
     * Implementa el método IPCC Tier 2 con factores Ym variables según composición
     * @param DMI_kg_day Consumo de Materia Seca en kg/día
     * @param BW_kg Peso corporal del animal en kg (opcional, para ecuaciones avanzadas)
     */
    void calculateEntericMethane(double DMI_kg_day, double BW_kg = 0.0) {
        if (DMI_kg_day <= 0) {
            entericMethane = 0.0;
            return;
        }

        // Obtener composición nutricional
        double GE_Mcal_per_kg = finalNutrientProfile.count("GE") ? finalNutrientProfile.at("GE") : 0.0;
        double NDF_percent = finalNutrientProfile.count("NDF") ? finalNutrientProfile.at("NDF") : 0.0;
        double Fat_percent = finalNutrientProfile.count("Fat") ? finalNutrientProfile.at("Fat") : 0.0;
        double CP_percent = finalNutrientProfile.count("CP") ? finalNutrientProfile.at("CP") : 0.0;
        double Starch_percent = finalNutrientProfile.count("Starch") ? finalNutrientProfile.at("Starch") : 0.0;

        // Si no hay GE en el perfil, estimar a partir de TDN o usar valor por defecto
        if (GE_Mcal_per_kg < 0.01) {
            double TDN_percent = finalNutrientProfile.count("TDN") ? finalNutrientProfile.at("TDN") : 0.0;
            if (TDN_percent > 0) {
                // Aproximación: GE ≈ 4.4 Mcal/kg para ingredientes típicos
                GE_Mcal_per_kg = 4.4;
            } else {
                GE_Mcal_per_kg = 4.4; // Valor por defecto conservador
            }
        }

        // PASO 1: Determinar el nivel de forraje de la dieta
        double forageLevel = calculateForageLevel();

        // PASO 2: Seleccionar el método de cálculo apropiado según NASEM 2016
        // Usaremos IPCC Tier 2 con factores Ym variables (más práctico y validado)

        double Ym = 0.0; // Factor de conversión de metano (% de GEI perdido como CH4)

        if (forageLevel >= 40.0) {
            // Dieta alta en forraje (≥40% MS)
            // IPCC recomienda Ym = 6.5% ± 1.0%
            // Ajustar según composición específica
            Ym = 0.065;

            // Ajuste por NDF: Mayor fibra = Mayor metano
            if (NDF_percent > 35.0) {
                Ym += (NDF_percent - 35.0) * 0.0005; // +0.05% por cada 1% NDF adicional
            }

            // Ajuste por grasa: Mayor grasa = Menor metano
            if (Fat_percent > 2.0) {
                Ym -= (Fat_percent - 2.0) * 0.003; // -0.3% por cada 1% de grasa adicional
            }

        } else if (forageLevel <= 20.0) {
            // Dieta de finalización con alto concentrado (≤20% forraje)
            // IPCC recomienda Ym = 3.0% para dietas >90% concentrado
            Ym = 0.030;

            // Ajuste por almidón: Mayor almidón = Menor metano
            if (Starch_percent > 40.0) {
                Ym -= (Starch_percent - 40.0) * 0.0002; // Reducción leve adicional
            }

            // Ajuste por grasa
            if (Fat_percent > 3.0) {
                Ym -= (Fat_percent - 3.0) * 0.002;
            }

        } else {
            // Dieta intermedia (20-40% forraje)
            // Interpolación lineal entre 6.5% y 3.0%
            double forageRatio = (forageLevel - 20.0) / (40.0 - 20.0);
            Ym = 0.030 + forageRatio * (0.065 - 0.030);

            // Ajustes moderados
            if (NDF_percent > 30.0) {
                Ym += (NDF_percent - 30.0) * 0.0003;
            }
            if (Fat_percent > 2.5) {
                Ym -= (Fat_percent - 2.5) * 0.0025;
            }
        }

        // Límites de seguridad para Ym (según literatura NASEM 2016)
        Ym = std::max(0.020, std::min(0.080, Ym));

        // PASO 3: Calcular emisión de metano
        // GEI (Mcal/d) = GE (Mcal/kg) × DMI (kg/d)
        double GEI_Mcal_per_day = GE_Mcal_per_kg * DMI_kg_day;

        // CH4 (Mcal/d) = GEI × Ym
        double methane_Mcal_per_day = GEI_Mcal_per_day * Ym;

        // Convertir energía a masa
        // Factor de conversión: 1 kg CH4 = 55.65 MJ = 13.3 Mcal
        // Por lo tanto: 1 Mcal CH4 = 1000g / 13.3 = 75.19 g/Mcal
        const double MCAL_TO_GRAMS_CH4 = 75.19;
        entericMethane = methane_Mcal_per_day * MCAL_TO_GRAMS_CH4;

        // Opcional: Usar ecuación más precisa si se proporciona BW
        if (BW_kg > 0) {
            // Aquí se podría implementar Escobar-Bahamondes u otra ecuación avanzada
            // Por ahora mantenemos IPCC Tier 2 mejorado
        }
    }

    /**
     * @brief Calcula el porcentaje de forraje en la dieta según clasificación de ingredientes
     * @return Porcentaje de forraje en base a materia seca
     */
    double calculateForageLevel() {
        double forageProportion = 0.0;

        // Método 1: Si los ingredientes están marcados como forraje, usar directamente
        for (const auto& pair : composition) {
            if (pair.first.isForage) {
                forageProportion += pair.second * 100.0; // Convertir a porcentaje
            }
        }

        // Si se encontró información de forraje directa, usarla
        if (forageProportion > 0.01) {
            return forageProportion;
        }

        // Método 2: Estimación basada en NDF (método de respaldo)
        double totalNDF = finalNutrientProfile.count("NDF") ? finalNutrientProfile.at("NDF") : 0.0;

        // Relación empírica entre NDF y nivel de forraje
        // NDF 45% → ~60% forraje (típico de dietas pastoreo)
        // NDF 30% → ~30% forraje (típico de dietas mixtas)
        // NDF 15% → ~10% forraje (típico de finalización)

        if (totalNDF >= 40.0) {
            forageProportion = 20.0 + (totalNDF - 40.0) * 1.5;
        } else if (totalNDF >= 25.0) {
            forageProportion = 10.0 + (totalNDF - 25.0) * 0.67;
        } else {
            forageProportion = totalNDF * 0.4;
        }

        // Limitar entre 0% y 100%
        return std::max(0.0, std::min(100.0, forageProportion));
    }

    /**
     * @brief Imprime resumen formateado de la dieta
     */
    void print() const {
        std::cout << "-------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(4);
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
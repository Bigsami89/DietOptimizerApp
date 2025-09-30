#include "ApiRequest.h"
#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

bool ApiRequest::fromJson(const std::string& jsonString, ApiRequest& request) {
    try {
        json j = json::parse(jsonString);

        // Animal requirements
        if (j.contains("animal_requirements")) {
            auto animalReq = j["animal_requirements"];

            if (animalReq.contains("DMI_kg_day")) {
                request.DMI_kg_day = animalReq["DMI_kg_day"];
            }

            // NUEVO: Leer peso corporal
            if (animalReq.contains("body_weight_kg")) {
                request.body_weight_kg = animalReq["body_weight_kg"];
            }

            if (animalReq.contains("min_requirements")) {
                auto minReq = animalReq["min_requirements"];
                for (auto& [key, value] : minReq.items()) {
                    request.minRequirements[key] = value;
                }
            }

            if (animalReq.contains("max_requirements")) {
                auto maxReq = animalReq["max_requirements"];
                for (auto& [key, value] : maxReq.items()) {
                    request.maxRequirements[key] = value;
                }
            }
        }

        // Available ingredients
        if (j.contains("available_ingredients")) {
            auto ingredients = j["available_ingredients"];
            for (auto& ingredientJson : ingredients) {
                std::string name = ingredientJson["name"];
                double cost = ingredientJson["cost"];

                // NUEVO: Leer clasificación de forraje
                bool isForage = false;
                if (ingredientJson.contains("is_forage")) {
                    isForage = ingredientJson["is_forage"];
                }

                std::map<std::string, double> nutrients;
                if (ingredientJson.contains("nutrients")) {
                    auto nutrientsJson = ingredientJson["nutrients"];
                    for (auto& [key, value] : nutrientsJson.items()) {
                        nutrients[key] = value;
                    }
                }

                // Crear ingrediente con clasificación de forraje
                Ingredient ing(name, cost, nutrients, isForage);
                request.availableIngredients.push_back(ing);
            }
        }

        return true;

    } catch (const std::exception& e) {
        std::cout << "JSON Parse Error: " << e.what() << std::endl;
        return false;
    }
}
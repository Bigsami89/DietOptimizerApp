#include "ApiRequest.h"
#include "json.hpp"  // Agregar esta línea
#include <iostream>

using json = nlohmann::json;

bool ApiRequest::fromJson(const std::string& jsonString, ApiRequest& request) {
    try {
        // Parse JSON con librería real
        json j = json::parse(jsonString);
        
        // Animal requirements
        if (j.contains("animal_requirements")) {
            auto animalReq = j["animal_requirements"];
            
            if (animalReq.contains("DMI_kg_day")) {
                request.DMI_kg_day = animalReq["DMI_kg_day"];
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
                
                std::map<std::string, double> nutrients;
                if (ingredientJson.contains("nutrients")) {
                    auto nutrientsJson = ingredientJson["nutrients"];
                    for (auto& [key, value] : nutrientsJson.items()) {
                        nutrients[key] = value;
                    }
                }
                
                request.availableIngredients.emplace_back(name, cost, nutrients);
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "JSON Parse Error: " << e.what() << std::endl;
        return false;
    }
}
package com.example.dietoptimizerapp.utils;

import com.example.dietoptimizerapp.models.AnimalType;
import com.example.dietoptimizerapp.models.Ingredient;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.List;
import java.util.Map;

public class JsonBuilder {

    public static String buildDietRequestJson(AnimalType animal, List<Ingredient> ingredients, String dietType) {
        try {
            JSONObject root = new JSONObject();

            // Animal requirements
            JSONObject animalReqs = new JSONObject();
            animalReqs.put("DMI_kg_day", animal.getDmiKgDay());
            animalReqs.put("body_weight_kg", animal.getBodyWeightKg()); // NUEVO

            // Adjust requirements based on diet type
            Map<String, Double> minReqs = adjustRequirementsForDietType(animal.getMinRequirements(), dietType, true);
            Map<String, Double> maxReqs = adjustRequirementsForDietType(animal.getMaxRequirements(), dietType, false);

            JSONObject minRequirements = new JSONObject();
            for (Map.Entry<String, Double> entry : minReqs.entrySet()) {
                minRequirements.put(entry.getKey(), entry.getValue());
            }

            JSONObject maxRequirements = new JSONObject();
            for (Map.Entry<String, Double> entry : maxReqs.entrySet()) {
                maxRequirements.put(entry.getKey(), entry.getValue());
            }

            animalReqs.put("min_requirements", minRequirements);
            animalReqs.put("max_requirements", maxRequirements);
            root.put("animal_requirements", animalReqs);

            // Available ingredients - MEJORADO con información de forraje
            JSONArray ingredientsArray = new JSONArray();
            for (Ingredient ingredient : ingredients) {
                if (ingredient.isSelected()) {
                    JSONObject ingObj = new JSONObject();
                    ingObj.put("name", ingredient.getName());
                    ingObj.put("cost", ingredient.getCost());
                    ingObj.put("is_forage", ingredient.isForage()); // NUEVO: Información crítica para metano

                    JSONObject nutrients = new JSONObject();
                    for (Map.Entry<String, Double> nutrient : ingredient.getNutrients().entrySet()) {
                        nutrients.put(nutrient.getKey(), nutrient.getValue());
                    }
                    ingObj.put("nutrients", nutrients);

                    ingredientsArray.put(ingObj);
                }
            }
            root.put("available_ingredients", ingredientsArray);

            return root.toString();

        } catch (Exception e) {
            e.printStackTrace();
            return "{}";
        }
    }

    private static Map<String, Double> adjustRequirementsForDietType(Map<String, Double> baseReqs, String dietType, boolean isMinimum) {
        Map<String, Double> adjustedReqs = new java.util.HashMap<>(baseReqs);

        switch (dietType.toLowerCase()) {
            case "crecimiento":
            case "growth":
                if (isMinimum) {
                    adjustedReqs.put("CP", adjustedReqs.getOrDefault("CP", 14.0) * 1.15);
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.10);
                }
                break;

            case "lactación":
            case "lactation":
                if (isMinimum) {
                    adjustedReqs.put("CP", adjustedReqs.getOrDefault("CP", 14.0) * 1.25);
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.20);
                    adjustedReqs.put("Ca", adjustedReqs.getOrDefault("Ca", 0.4) * 1.40);
                }
                break;

            case "engorde":
            case "fattening":
                if (isMinimum) {
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.15);
                } else {
                    adjustedReqs.put("NDF", adjustedReqs.getOrDefault("NDF", 35.0) * 0.85);
                }
                break;

            case "mantenimiento":
            case "maintenance":
            default:
                // No adjustments needed
                break;
        }

        return adjustedReqs;
    }
}
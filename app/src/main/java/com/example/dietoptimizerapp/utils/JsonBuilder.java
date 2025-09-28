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

            // Available ingredients - CORREGIR AQUÍ
            JSONArray ingredientsArray = new JSONArray();
            for (Ingredient ingredient : ingredients) {
                // CAMBIAR: usar selectedIngredients o verificar isSelected()
                if (ingredient.isSelected()) {  // Solo ingredientes seleccionados
                    JSONObject ingObj = new JSONObject();
                    ingObj.put("name", ingredient.getName());
                    ingObj.put("cost", ingredient.getCost());

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
        // Clone the map
        Map<String, Double> adjustedReqs = new java.util.HashMap<>(baseReqs);

        // Adjust based on diet type
        switch (dietType.toLowerCase()) {
            case "growth":
                if (isMinimum) {
                    adjustedReqs.put("CP", adjustedReqs.getOrDefault("CP", 14.0) * 1.15); // +15% proteína
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.10); // +10% energía
                }
                break;

            case "lactation":
                if (isMinimum) {
                    adjustedReqs.put("CP", adjustedReqs.getOrDefault("CP", 14.0) * 1.25); // +25% proteína
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.20); // +20% energía
                    adjustedReqs.put("Ca", adjustedReqs.getOrDefault("Ca", 0.4) * 1.40); // +40% calcio
                }
                break;

            case "fattening":
                if (isMinimum) {
                    adjustedReqs.put("NEm", adjustedReqs.getOrDefault("NEm", 1.6) * 1.15); // +15% energía
                } else {
                    adjustedReqs.put("NDF", adjustedReqs.getOrDefault("NDF", 35.0) * 0.85); // -15% fibra
                }
                break;

            case "maintenance":
            default:
                // No adjustments needed
                break;
        }

        return adjustedReqs;
    }
}
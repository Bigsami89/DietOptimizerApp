package com.example.dietoptimizerapp.models;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Ingredient {
    private String id;
    private String name;
    private String localName;
    private String category;
    private double cost; // Precio por kg
    private String costUnit;
    private Map<String, Double> nutrients;
    private Map<String, Double> nutrientVariability;
    private boolean selected;
    private boolean isForage; // NUEVO: Clasificación para cálculo de metano
    private String availability;
    private String storageRequirements;
    private List<String> antiNutritionalFactors;
    private int palatabilityScore;
    private List<String> mixingRestrictions;
    private double minInclusion;
    private double maxInclusion;
    private double recommendedInclusion;
    private String description;
    private List<String> images;
    private List<String> references;
    private String notes;
    private boolean isCustom;

    public Ingredient(String name, double cost) {
        this(null, name, cost);
    }

    // Constructor con nutrientes
    public Ingredient(String name, double cost, Map<String, Double> nutrients) {
        this(null, name, cost, nutrients);
    }

    public Ingredient(String id, String name, double cost) {
        this(id, name, cost, new HashMap<>());
    }

    public Ingredient(String id, String name, double cost, Map<String, Double> nutrients) {
        this.id = id;
        this.name = name;
        this.localName = name;
        this.category = "general";
        this.cost = cost;
        this.costUnit = "USD/kg";
        this.nutrients = nutrients != null ? new HashMap<>(nutrients) : new HashMap<>();
        this.nutrientVariability = new HashMap<>();
        this.selected = false;
        this.isForage = false;
        this.availability = "common";
        this.storageRequirements = "";
        this.antiNutritionalFactors = new ArrayList<>();
        this.palatabilityScore = 0;
        this.mixingRestrictions = new ArrayList<>();
        this.minInclusion = 0.0;
        this.maxInclusion = 1.0;
        this.recommendedInclusion = 0.0;
        this.description = "";
        this.images = new ArrayList<>();
        this.references = new ArrayList<>();
        this.notes = "";
        this.isCustom = false;
        autoDetectForage(); // Detectar automáticamente basado en composición
    }

    // Getters y setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getLocalName() { return localName; }
    public void setLocalName(String localName) { this.localName = localName; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public double getCost() { return cost; }
    public void setCost(double cost) { this.cost = cost; }

    public String getCostUnit() { return costUnit; }
    public void setCostUnit(String costUnit) { this.costUnit = costUnit; }

    public Map<String, Double> getNutrients() { return nutrients; }
    public void setNutrients(Map<String, Double> nutrients) {
        this.nutrients = nutrients;
        autoDetectForage(); // Re-detectar cuando cambian los nutrientes
    }

    public Map<String, Double> getNutrientVariability() { return nutrientVariability; }
    public void setNutrientVariability(Map<String, Double> nutrientVariability) {
        this.nutrientVariability = nutrientVariability != null ? new HashMap<>(nutrientVariability) : new HashMap<>();
    }

    public boolean isSelected() { return selected; }
    public void setSelected(boolean selected) { this.selected = selected; }

    // NUEVOS métodos para clasificación de forraje
    public boolean isForage() { return isForage; }
    public void setForage(boolean isForage) { this.isForage = isForage; }

    public void addNutrient(String nutrient, double value) {
        this.nutrients.put(nutrient, value);
    }

    public double getNutrient(String nutrient) {
        return nutrients.getOrDefault(nutrient, 0.0);
    }

    public String getAvailability() { return availability; }
    public void setAvailability(String availability) { this.availability = availability; }

    public String getStorageRequirements() { return storageRequirements; }
    public void setStorageRequirements(String storageRequirements) { this.storageRequirements = storageRequirements; }

    public List<String> getAntiNutritionalFactors() { return Collections.unmodifiableList(antiNutritionalFactors); }
    public void setAntiNutritionalFactors(List<String> antiNutritionalFactors) {
        this.antiNutritionalFactors = antiNutritionalFactors != null ? new ArrayList<>(antiNutritionalFactors) : new ArrayList<>();
    }

    public int getPalatabilityScore() { return palatabilityScore; }
    public void setPalatabilityScore(int palatabilityScore) { this.palatabilityScore = palatabilityScore; }

    public List<String> getMixingRestrictions() { return Collections.unmodifiableList(mixingRestrictions); }
    public void setMixingRestrictions(List<String> mixingRestrictions) {
        this.mixingRestrictions = mixingRestrictions != null ? new ArrayList<>(mixingRestrictions) : new ArrayList<>();
    }

    public double getMinInclusion() { return minInclusion; }
    public void setMinInclusion(double minInclusion) { this.minInclusion = minInclusion; }

    public double getMaxInclusion() { return maxInclusion; }
    public void setMaxInclusion(double maxInclusion) { this.maxInclusion = maxInclusion; }

    public double getRecommendedInclusion() { return recommendedInclusion; }
    public void setRecommendedInclusion(double recommendedInclusion) { this.recommendedInclusion = recommendedInclusion; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public List<String> getImages() { return Collections.unmodifiableList(images); }
    public void setImages(List<String> images) {
        this.images = images != null ? new ArrayList<>(images) : new ArrayList<>();
    }

    public List<String> getReferences() { return Collections.unmodifiableList(references); }
    public void setReferences(List<String> references) {
        this.references = references != null ? new ArrayList<>(references) : new ArrayList<>();
    }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }

    public boolean isCustom() { return isCustom; }
    public void setCustom(boolean custom) { isCustom = custom; }

    /**
     * Detecta automáticamente si un ingrediente es forraje según criterios NASEM 2016
     * Criterios: NDF > 25%, CP < 20%, TDN < 75%
     */
    private void autoDetectForage() {
        double ndf = getNutrient("NDF");
        double cp = getNutrient("CP");
        double tdn = getNutrient("TDN");

        // Lógica de detección según composición nutricional
        if (ndf > 25.0 && cp < 20.0) {
            this.isForage = true;
        } else if (ndf > 35.0) {
            // NDF muy alto casi siempre indica forraje
            this.isForage = true;
        } else if (tdn < 65.0 && ndf > 20.0) {
            // Baja digestibilidad con fibra moderada
            this.isForage = true;
        } else {
            this.isForage = false;
        }
    }

    // Cálculos de costo
    public double getCostPerTonne() {
        return cost * 1000;
    }

    @Override
    public String toString() {
        String type = isForage ? " [Forraje]" : " [Concentrado]";
        return name + type + " ($" + String.format("%.3f", cost) + "/kg)";
    }
}
package com.example.dietoptimizerapp.models;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AnimalType {
    private String id;
    private String name;
    private String species;
    private String breed;
    private String productionType;
    private String lifeStage;
    private double dmiKgDay; // Dry Matter Intake kg/day
    private double bodyWeightKg; // NUEVO: Peso corporal en kg para cálculos de metano
    private double[] bodyWeightRange;
    private String dmiFormula;
    private Map<String, Double> minRequirements;
    private Map<String, Double> maxRequirements;
    private Map<String, RequirementRange> optimalRanges;
    private Map<String, Double> dietAdjustments;
    private List<String> healthConsiderations;
    private Map<String, Double> productivityTargets;
    private Map<String, Double> environmentalFactors;
    private String image;
    private String description;
    private String lastUpdated;

    public AnimalType(String name, double dmiKgDay, double bodyWeightKg) {
        this(null, name, dmiKgDay, bodyWeightKg);
    }

    public AnimalType(String id, String name, double dmiKgDay, double bodyWeightKg) {
        this.id = id;
        this.name = name;
        this.species = "cattle";
        this.breed = "";
        this.productionType = "mixed";
        this.lifeStage = "adult";
        this.dmiKgDay = dmiKgDay;
        this.bodyWeightKg = bodyWeightKg;
        this.bodyWeightRange = new double[]{bodyWeightKg, bodyWeightKg};
        this.dmiFormula = "";
        this.minRequirements = new HashMap<>();
        this.maxRequirements = new HashMap<>();
        this.optimalRanges = new HashMap<>();
        this.dietAdjustments = new HashMap<>();
        this.healthConsiderations = new ArrayList<>();
        this.productivityTargets = new HashMap<>();
        this.environmentalFactors = new HashMap<>();
        this.image = "";
        this.description = "";
        this.lastUpdated = "";
    }

    // Getters y setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getSpecies() { return species; }
    public void setSpecies(String species) { this.species = species; }

    public String getBreed() { return breed; }
    public void setBreed(String breed) { this.breed = breed; }

    public String getProductionType() { return productionType; }
    public void setProductionType(String productionType) { this.productionType = productionType; }

    public String getLifeStage() { return lifeStage; }
    public void setLifeStage(String lifeStage) { this.lifeStage = lifeStage; }

    public double getDmiKgDay() { return dmiKgDay; }
    public void setDmiKgDay(double dmiKgDay) { this.dmiKgDay = dmiKgDay; }

    public double getBodyWeightKg() { return bodyWeightKg; }
    public void setBodyWeightKg(double bodyWeightKg) { this.bodyWeightKg = bodyWeightKg; }

    public double[] getBodyWeightRange() { return bodyWeightRange; }
    public void setBodyWeightRange(double[] bodyWeightRange) { this.bodyWeightRange = bodyWeightRange; }

    public String getDmiFormula() { return dmiFormula; }
    public void setDmiFormula(String dmiFormula) { this.dmiFormula = dmiFormula; }

    public Map<String, Double> getMinRequirements() { return minRequirements; }
    public void setMinRequirements(Map<String, Double> minRequirements) { this.minRequirements = minRequirements; }

    public Map<String, Double> getMaxRequirements() { return maxRequirements; }
    public void setMaxRequirements(Map<String, Double> maxRequirements) { this.maxRequirements = maxRequirements; }

    public Map<String, RequirementRange> getOptimalRanges() { return Collections.unmodifiableMap(optimalRanges); }
    public void setOptimalRanges(Map<String, RequirementRange> optimalRanges) { this.optimalRanges = optimalRanges != null ? new HashMap<>(optimalRanges) : new HashMap<>(); }

    public Map<String, Double> getDietAdjustments() { return dietAdjustments; }
    public void setDietAdjustments(Map<String, Double> dietAdjustments) { this.dietAdjustments = dietAdjustments != null ? new HashMap<>(dietAdjustments) : new HashMap<>(); }

    public List<String> getHealthConsiderations() { return Collections.unmodifiableList(healthConsiderations); }
    public void setHealthConsiderations(List<String> healthConsiderations) { this.healthConsiderations = healthConsiderations != null ? new ArrayList<>(healthConsiderations) : new ArrayList<>(); }

    public Map<String, Double> getProductivityTargets() { return productivityTargets; }
    public void setProductivityTargets(Map<String, Double> productivityTargets) { this.productivityTargets = productivityTargets != null ? new HashMap<>(productivityTargets) : new HashMap<>(); }

    public Map<String, Double> getEnvironmentalFactors() { return environmentalFactors; }
    public void setEnvironmentalFactors(Map<String, Double> environmentalFactors) { this.environmentalFactors = environmentalFactors != null ? new HashMap<>(environmentalFactors) : new HashMap<>(); }

    public String getImage() { return image; }
    public void setImage(String image) { this.image = image; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getLastUpdated() { return lastUpdated; }
    public void setLastUpdated(String lastUpdated) { this.lastUpdated = lastUpdated; }

    public void addMinRequirement(String nutrient, double value) {
        this.minRequirements.put(nutrient, value);
    }

    public void addMaxRequirement(String nutrient, double value) {
        this.maxRequirements.put(nutrient, value);
    }

    public void addOptimalRange(String nutrient, double min, double max) {
        this.optimalRanges.put(nutrient, new RequirementRange(min, max));
    }

    public void addHealthConsideration(String consideration) {
        this.healthConsiderations.add(consideration);
    }

    public void addProductivityTarget(String metric, double value) {
        this.productivityTargets.put(metric, value);
    }

    public void addEnvironmentalFactor(String metric, double value) {
        this.environmentalFactors.put(metric, value);
    }

    public String getReadableLastUpdated() {
        return lastUpdated == null ? "" : lastUpdated;
    }

    @Override
    public String toString() {
        return name + " (DMI: " + dmiKgDay + " kg/día, " + bodyWeightKg + " kg)";
    }

    // Factory method con datos científicamente actualizados
    public static AnimalType[] getDefaultAnimalTypes() {
        // Bovino Lechero: ~650 kg, 22 kg MS/día
        AnimalType bovinoLechero = new AnimalType("Bovino Lechero", 22.0, 650.0);
        bovinoLechero.addMinRequirement("CP", 16.5);
        bovinoLechero.addMinRequirement("NEm", 1.65);
        bovinoLechero.addMinRequirement("Ca", 0.30);
        bovinoLechero.addMaxRequirement("NDF", 35.0);
        bovinoLechero.addMinRequirement("P", 0.38);
        bovinoLechero.addMaxRequirement("CP", 18.5);

        // Bovino de Carne: ~450 kg, 10.5 kg MS/día
        AnimalType bovinoCarne = new AnimalType("Bovino de Carne", 10.5, 450.0);
        bovinoCarne.addMinRequirement("CP", 12.0);
        bovinoCarne.addMinRequirement("NEm", 1.9);
        bovinoCarne.addMinRequirement("Ca", 0.30);
        bovinoCarne.addMaxRequirement("NDF", 30.0);
        bovinoCarne.addMinRequirement("P", 0.25);
        bovinoCarne.addMaxRequirement("CP", 14.0);

        // Ovino: ~70 kg, 1.8 kg MS/día
        AnimalType ovino = new AnimalType("Ovino", 1.8, 70.0);
        ovino.addMinRequirement("CP", 11.0);
        ovino.addMinRequirement("NEm", 1.4);
        ovino.addMinRequirement("Ca", 0.30);
        ovino.addMaxRequirement("NDF", 50.0);
        ovino.addMinRequirement("P", 0.22);
        ovino.addMaxRequirement("CP", 15.0);

        // Caprino: ~60 kg, 2.2 kg MS/día
        AnimalType caprino = new AnimalType("Caprino", 2.2, 60.0);
        caprino.addMinRequirement("CP", 14.0);
        caprino.addMinRequirement("NEm", 1.65);
        caprino.addMinRequirement("Ca", 0.60);
        caprino.addMaxRequirement("NDF", 40.0);
        caprino.addMinRequirement("P", 0.35);
        caprino.addMaxRequirement("CP", 18.0);

        return new AnimalType[]{bovinoLechero, bovinoCarne, ovino, caprino};
    }

    public static class RequirementRange {
        private final double min;
        private final double max;

        public RequirementRange(double min, double max) {
            this.min = min;
            this.max = max;
        }

        public double getMin() {
            return min;
        }

        public double getMax() {
            return max;
        }
    }
}
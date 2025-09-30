package com.example.dietoptimizerapp.models;

import java.util.HashMap;
import java.util.Map;

public class AnimalType {
    private String name;
    private double dmiKgDay; // Dry Matter Intake kg/day
    private double bodyWeightKg; // NUEVO: Peso corporal en kg para cálculos de metano
    private Map<String, Double> minRequirements;
    private Map<String, Double> maxRequirements;

    public AnimalType(String name, double dmiKgDay, double bodyWeightKg) {
        this.name = name;
        this.dmiKgDay = dmiKgDay;
        this.bodyWeightKg = bodyWeightKg;
        this.minRequirements = new HashMap<>();
        this.maxRequirements = new HashMap<>();
    }

    // Getters y setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public double getDmiKgDay() { return dmiKgDay; }
    public void setDmiKgDay(double dmiKgDay) { this.dmiKgDay = dmiKgDay; }

    public double getBodyWeightKg() { return bodyWeightKg; }
    public void setBodyWeightKg(double bodyWeightKg) { this.bodyWeightKg = bodyWeightKg; }

    public Map<String, Double> getMinRequirements() { return minRequirements; }
    public void setMinRequirements(Map<String, Double> minRequirements) { this.minRequirements = minRequirements; }

    public Map<String, Double> getMaxRequirements() { return maxRequirements; }
    public void setMaxRequirements(Map<String, Double> maxRequirements) { this.maxRequirements = maxRequirements; }

    public void addMinRequirement(String nutrient, double value) {
        this.minRequirements.put(nutrient, value);
    }

    public void addMaxRequirement(String nutrient, double value) {
        this.maxRequirements.put(nutrient, value);
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
}
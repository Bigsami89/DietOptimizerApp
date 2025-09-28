package com.example.dietoptimizerapp.models;

import java.util.HashMap;
import java.util.Map;

public class AnimalType {
    private String name;
    private double dmiKgDay; // Dry Matter Intake kg/day
    private Map<String, Double> minRequirements;
    private Map<String, Double> maxRequirements;

    public AnimalType(String name, double dmiKgDay) {
        this.name = name;
        this.dmiKgDay = dmiKgDay;
        this.minRequirements = new HashMap<>();
        this.maxRequirements = new HashMap<>();
    }

    // Getters y setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public double getDmiKgDay() { return dmiKgDay; }
    public void setDmiKgDay(double dmiKgDay) { this.dmiKgDay = dmiKgDay; }

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
        return name + " (DMI: " + dmiKgDay + " kg/día)";
    }

    // Factory method para crear tipos de animales predefinidos
    public static AnimalType[] getDefaultAnimalTypes() {
        // --- Requerimientos para Bovino Lechero ---
// Valores ajustados para una vaca de ~650 kg produciendo ~25 L/día.
// El consumo de materia seca (CMS) se estima en 22 kg.
        AnimalType bovinoLechero = new AnimalType("Bovino Lechero", 22.0);
        bovinoLechero.addMinRequirement("CP", 16.5); // % - Ligero aumento para sostener la producción de leche.
        bovinoLechero.addMinRequirement("NEm", 1.65); // Mcal/kg MS - Nivel de energía para lactancia.
        bovinoLechero.addMinRequirement("Ca", 0.75); // % - Mayor demanda por la producción de leche.
        bovinoLechero.addMaxRequirement("NDF", 35.0); // % - Un límite máximo ligeramente superior es más realista.
        bovinoLechero.addMinRequirement("P", 0.38); // % - Se define como mínimo en lugar de máximo. El fósforo es esencial.
        bovinoLechero.addMaxRequirement("CP", 18.5); // % - Límite superior común para vacas de alta producción.

// --- Requerimientos para Bovino de Carne ---
// Ajustado para un animal en etapa de finalización/engorde (~450 kg).
// El consumo de materia seca (CMS) se estima en 10.5 kg.
        AnimalType bovinoCarne = new AnimalType("Bovino de Carne", 10.5);
        bovinoCarne.addMinRequirement("CP", 12.0); // % - Reducido, ya que en finalización la demanda de proteína es menor.
        bovinoCarne.addMinRequirement("NEm", 1.9); // Mcal/kg MS - Mayor densidad energética para la ganancia de peso.
        bovinoCarne.addMinRequirement("Ca", 0.30); // % - Requerimientos menores que en lecheros.
        bovinoCarne.addMaxRequirement("NDF", 30.0); // % - Se busca menor fibra para maximizar el consumo de energía.
        bovinoCarne.addMinRequirement("P", 0.25); // % - Definido como mínimo.
        bovinoCarne.addMaxRequirement("CP", 14.0); // % - Un exceso de proteína en esta etapa no es costo-eficiente.

// --- Requerimientos para Ovino ---
// Ajustado para una oveja adulta en mantenimiento (~70 kg).
// El consumo de materia seca (CMS) se estima en 1.8 kg.
        AnimalType ovino = new AnimalType("Ovino", 1.8);
        ovino.addMinRequirement("CP", 11.0); // % - Adecuado para mantenimiento.
        ovino.addMinRequirement("NEm", 1.4); // Mcal/kg MS - Nivel de energía de mantenimiento.
        ovino.addMinRequirement("Ca", 0.30); // % - Valor estándar.
        ovino.addMaxRequirement("NDF", 50.0); // % - Los ovinos pueden manejar niveles más altos de fibra.
        ovino.addMinRequirement("P", 0.22); // % - Definido como mínimo.
        ovino.addMaxRequirement("CP", 15.0); // %

// --- Requerimientos para Caprino ---
// Ajustado para una cabra lechera en producción media (~60 kg).
// El consumo de materia seca (CMS) se estima en 2.2 kg.
        AnimalType caprino = new AnimalType("Caprino", 2.2);
        caprino.addMinRequirement("CP", 14.0); // % - Necesario para producción de leche.
        caprino.addMinRequirement("NEm", 1.65); // Mcal/kg MS - Similar a vacas lecheras de menor producción.
        caprino.addMinRequirement("Ca", 0.60); // %
        caprino.addMaxRequirement("NDF", 40.0); // %
        caprino.addMinRequirement("P", 0.35); // % - Definido como mínimo.
        caprino.addMaxRequirement("CP", 18.0); // %

        return new AnimalType[]{bovinoLechero, bovinoCarne, ovino, caprino};
    }
}
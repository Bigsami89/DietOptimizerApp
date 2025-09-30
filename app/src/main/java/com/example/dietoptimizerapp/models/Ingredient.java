package com.example.dietoptimizerapp.models;

import java.util.HashMap;
import java.util.Map;

public class Ingredient {
    private String name;
    private double cost; // Precio por kg
    private Map<String, Double> nutrients;
    private boolean selected;
    private boolean isForage; // NUEVO: Clasificación para cálculo de metano

    public Ingredient(String name, double cost) {
        this.name = name;
        this.cost = cost;
        this.nutrients = new HashMap<>();
        this.selected = false;
        this.isForage = false;
        autoDetectForage(); // Detectar automáticamente
    }

    // Constructor con nutrientes
    public Ingredient(String name, double cost, Map<String, Double> nutrients) {
        this.name = name;
        this.cost = cost;
        this.nutrients = nutrients;
        this.selected = false;
        this.isForage = false;
        autoDetectForage(); // Detectar automáticamente basado en composición
    }

    // Getters y setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public double getCost() { return cost; }
    public void setCost(double cost) { this.cost = cost; }

    public Map<String, Double> getNutrients() { return nutrients; }
    public void setNutrients(Map<String, Double> nutrients) {
        this.nutrients = nutrients;
        autoDetectForage(); // Re-detectar cuando cambian los nutrientes
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
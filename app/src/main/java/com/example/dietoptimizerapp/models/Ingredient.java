package com.example.dietoptimizerapp.models;

import java.util.HashMap;
import java.util.Map;

public class Ingredient {
    private String name;
    private double cost; // Precio por kg
    private Map<String, Double> nutrients;
    private boolean selected;

    public Ingredient(String name, double cost) {
        this.name = name;
        this.cost = cost;
        this.nutrients = new HashMap<>();
        this.selected = false;
    }

    // Constructor con nutrientes
    public Ingredient(String name, double cost, Map<String, Double> nutrients) {
        this.name = name;
        this.cost = cost;
        this.nutrients = nutrients;
        this.selected = false;
    }

    // Getters y setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public double getCost() { return cost; }
    public void setCost(double cost) { this.cost = cost; }

    public Map<String, Double> getNutrients() { return nutrients; }
    public void setNutrients(Map<String, Double> nutrients) { this.nutrients = nutrients; }

    public boolean isSelected() { return selected; }
    public void setSelected(boolean selected) { this.selected = selected; }

    public void addNutrient(String nutrient, double value) {
        this.nutrients.put(nutrient, value);
    }

    public double getNutrient(String nutrient) {
        return nutrients.getOrDefault(nutrient, 0.0);
    }

    // Cálculos de costo
    public double getCostPerTonne() {
        return cost * 1000;
    }

    @Override
    public String toString() {
        return name + " ($" + String.format("%.3f", cost) + "/kg)";
    }
}
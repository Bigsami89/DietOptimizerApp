package com.example.dietoptimizerapp.models;

import java.util.ArrayList;

public class DietResult {
    public ArrayList<DietComponent> components;
    public double totalCost; // Costo por kg
    public double totalMethane; // Emisiones en g/día

    public DietResult() {
        components = new ArrayList<DietComponent>();
        totalCost = 0.0;
        totalMethane = 0.0;
    }

    // Cálculos derivados
    public double getCostPerTonne() {
        return totalCost * 1000;
    }

    public double getDailyCost(double dmiKgDay, int quantity) {
        return totalCost * dmiKgDay * quantity;
    }

    public double getMonthlyCost(double dmiKgDay, int quantity) {
        return getDailyCost(dmiKgDay, quantity) * 30;
    }

    public double getEmissionsPerTonne(double dmiKgDay) {
        return (totalMethane / dmiKgDay) * 1000;
    }

    public double getDailyEmissions(int quantity) {
        return totalMethane * quantity;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Costo: $").append(String.format("%.3f", totalCost)).append("/kg\n");
        sb.append("Metano: ").append(String.format("%.1f", totalMethane)).append(" g/día\n");

        for (DietComponent comp : components) {
            sb.append("- ").append(comp.ingredientName)
                    .append(": ").append(String.format("%.1f", comp.proportion * 100)).append("%\n");
        }
        return sb.toString();
    }
}
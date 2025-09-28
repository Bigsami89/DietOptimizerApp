package com.example.dietoptimizerapp.utils;

import com.example.dietoptimizerapp.models.DietResult;

public class CostCalculator {

    public static class CostBreakdown {
        public double costPerKg;
        public double costPerTonne;
        public double dailyCost;
        public double monthlyCost;
        public double annualCost;

        public double emissionsPerDay;
        public double emissionsPerTonne;
        public double emissionsPerMonth;

        public CostBreakdown(double costPerKg, double costPerTonne, double dailyCost,
                             double monthlyCost, double annualCost, double emissionsPerDay,
                             double emissionsPerTonne, double emissionsPerMonth) {
            this.costPerKg = costPerKg;
            this.costPerTonne = costPerTonne;
            this.dailyCost = dailyCost;
            this.monthlyCost = monthlyCost;
            this.annualCost = annualCost;
            this.emissionsPerDay = emissionsPerDay;
            this.emissionsPerTonne = emissionsPerTonne;
            this.emissionsPerMonth = emissionsPerMonth;
        }
    }

    public static CostBreakdown calculateCosts(DietResult diet, double dmiKgDay, int quantity) {
        double costPerKg = diet.totalCost;
        double costPerTonne = costPerKg * 1000;

        double dailyWeight = dmiKgDay * quantity;
        double dailyCost = costPerKg * dailyWeight;
        double monthlyCost = dailyCost * 30;
        double annualCost = dailyCost * 365;

        double emissionsPerDay = diet.totalMethane * quantity;
        double emissionsPerTonne = (diet.totalMethane / dmiKgDay) * 1000;
        double emissionsPerMonth = emissionsPerDay * 30;

        return new CostBreakdown(
                costPerKg, costPerTonne, dailyCost, monthlyCost, annualCost,
                emissionsPerDay, emissionsPerTonne, emissionsPerMonth
        );
    }

    public static String formatCurrency(double amount) {
        return "$" + String.format("%.2f", amount);
    }

    public static String formatWeight(double grams) {
        if (grams >= 1000) {
            return String.format("%.2f", grams / 1000) + " kg";
        }
        return String.format("%.1f", grams) + " g";
    }
}
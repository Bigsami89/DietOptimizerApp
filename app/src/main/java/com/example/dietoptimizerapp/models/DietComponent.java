package com.example.dietoptimizerapp.models;

public class DietComponent {
    public String ingredientName;
    public double proportion;

    public DietComponent() {
        this.ingredientName = "";
        this.proportion = 0.0;
    }

    public DietComponent(String ingredientName, double proportion) {
        this.ingredientName = ingredientName;
        this.proportion = proportion;
    }

    // Getters y setters
    public String getIngredientName() {
        return ingredientName;
    }

    public void setIngredientName(String ingredientName) {
        this.ingredientName = ingredientName;
    }

    public double getProportion() {
        return proportion;
    }

    public void setProportion(double proportion) {
        this.proportion = proportion;
    }

    // Métodos utilitarios
    public double getPercentage() {
        return proportion * 100.0;
    }

    public String getFormattedPercentage() {
        return String.format("%.1f%%", getPercentage());
    }

    @Override
    public String toString() {
        return ingredientName + ": " + getFormattedPercentage();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        DietComponent that = (DietComponent) obj;
        return Double.compare(that.proportion, proportion) == 0 &&
                ingredientName.equals(that.ingredientName);
    }

    @Override
    public int hashCode() {
        int result = ingredientName.hashCode();
        long temp = Double.doubleToLongBits(proportion);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}
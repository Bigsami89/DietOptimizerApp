package com.example.dietoptimizerapp.adapters;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.example.dietoptimizerapp.R;
import com.example.dietoptimizerapp.models.DietComponent;
import com.example.dietoptimizerapp.models.DietResult;
import com.example.dietoptimizerapp.utils.CostCalculator;
import java.util.List;

public class ResultsAdapter extends RecyclerView.Adapter<ResultsAdapter.ResultViewHolder> {

    public static class DietResultWithCosts {
        public DietResult diet;
        public CostCalculator.CostBreakdown costs;
        public String title;

        public DietResultWithCosts(DietResult diet, CostCalculator.CostBreakdown costs, String title) {
            this.diet = diet;
            this.costs = costs;
            this.title = title;
        }
    }

    private List<DietResultWithCosts> results;

    public ResultsAdapter(List<DietResultWithCosts> results) {
        this.results = results;
    }

    @NonNull
    @Override
    public ResultViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_diet_result, parent, false);
        return new ResultViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ResultViewHolder holder, int position) {
        DietResultWithCosts item = results.get(position);
        holder.bind(item);
    }

    @Override
    public int getItemCount() {
        return results.size();
    }

    public void updateResults(List<DietResultWithCosts> newResults) {
        this.results = newResults;
        notifyDataSetChanged();
    }

    static class ResultViewHolder extends RecyclerView.ViewHolder {
        private TextView dietTitleText;
        private TextView dietBadgeText;
        private LinearLayout compositionLayout;
        private TextView costPerKgText;
        private TextView costPerTonneText;
        private TextView dailyCostText;
        private TextView quantityText;
        private TextView emissionsPerTonneText;
        private TextView dailyEmissionsText;
        private TextView monthlyCostText;

        public ResultViewHolder(@NonNull View itemView) {
            super(itemView);

            dietTitleText = itemView.findViewById(R.id.dietTitleText);
            dietBadgeText = itemView.findViewById(R.id.dietBadgeText);
            compositionLayout = itemView.findViewById(R.id.compositionLayout);
            costPerKgText = itemView.findViewById(R.id.costPerKgText);
            costPerTonneText = itemView.findViewById(R.id.costPerTonneText);
            dailyCostText = itemView.findViewById(R.id.dailyCostText);
            quantityText = itemView.findViewById(R.id.quantityText);
            emissionsPerTonneText = itemView.findViewById(R.id.emissionsPerTonneText);
            dailyEmissionsText = itemView.findViewById(R.id.dailyEmissionsText);
            monthlyCostText = itemView.findViewById(R.id.monthlyCostText);
        }

        public void bind(DietResultWithCosts item) {
            // Set title and badge
            dietTitleText.setText(item.title);

            String badge = "";
            if (item.title.contains("Mejor Costo")) badge = "Mejor Costo";
            else if (item.title.contains("Balance")) badge = "Balance";
            else badge = "Menos Emisiones";
            dietBadgeText.setText(badge);

            // Clear and populate composition
            compositionLayout.removeAllViews();
            for (DietComponent component : item.diet.components) {
                View compView = LayoutInflater.from(itemView.getContext())
                        .inflate(R.layout.item_composition_component, compositionLayout, false);

                TextView nameText = compView.findViewById(R.id.componentNameText);
                TextView percentText = compView.findViewById(R.id.componentPercentText);

                nameText.setText(component.ingredientName);
                percentText.setText(String.format("%.1f%%", component.proportion * 100));

                compositionLayout.addView(compView);
            }

            // Set cost and emissions
            costPerKgText.setText(CostCalculator.formatCurrency(item.costs.costPerKg));
            costPerTonneText.setText(CostCalculator.formatCurrency(item.costs.costPerTonne));
            dailyCostText.setText(CostCalculator.formatCurrency(item.costs.dailyCost));

            // Calculate quantity from daily cost and cost per kg
            int quantity = (int)Math.round(item.costs.dailyCost / (item.costs.costPerKg * 10.5)); // Assuming 10.5 kg DMI
            quantityText.setText(quantity + " animal" + (quantity > 1 ? "es" : ""));

            emissionsPerTonneText.setText(String.format("%.1f", item.costs.emissionsPerTonne));
            dailyEmissionsText.setText(String.format("%.1f g CH₄", item.costs.emissionsPerDay));
            monthlyCostText.setText(CostCalculator.formatCurrency(item.costs.monthlyCost));
        }
    }
}
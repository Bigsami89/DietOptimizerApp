package com.example.dietoptimizerapp.adapters;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.example.dietoptimizerapp.R;
import com.example.dietoptimizerapp.models.Ingredient;
import java.util.ArrayList;
import java.util.List;

public class IngredientSelectionAdapter extends RecyclerView.Adapter<IngredientSelectionAdapter.IngredientViewHolder> {

    public interface OnIngredientSelectionListener {
        void onIngredientSelected(Ingredient ingredient, boolean isSelected);
        void onIngredientInfoClicked(Ingredient ingredient);
    }

    private List<Ingredient> ingredients;
    private List<Ingredient> filteredIngredients;
    private OnIngredientSelectionListener listener;

    public IngredientSelectionAdapter(List<Ingredient> ingredients, OnIngredientSelectionListener listener) {
        this.ingredients = ingredients;
        this.filteredIngredients = new ArrayList<>(ingredients);
        this.listener = listener;
    }

    @NonNull
    @Override
    public IngredientViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_ingredient_selection, parent, false);
        return new IngredientViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull IngredientViewHolder holder, int position) {
        Ingredient ingredient = filteredIngredients.get(position);
        holder.bind(ingredient, listener);
    }

    @Override
    public int getItemCount() {
        return filteredIngredients.size();
    }

    /**
     * Actualiza la lista de ingredientes
     */
    public void updateIngredients(List<Ingredient> newIngredients) {
        this.ingredients = newIngredients;
        this.filteredIngredients = new ArrayList<>(newIngredients);
        notifyDataSetChanged();
    }

    /**
     * Filtra ingredientes por nombre
     */
    public void filter(String query) {
        filteredIngredients.clear();

        if (query.isEmpty()) {
            filteredIngredients.addAll(ingredients);
        } else {
            String lowerCaseQuery = query.toLowerCase().trim();
            for (Ingredient ingredient : ingredients) {
                if (ingredient.getName().toLowerCase().contains(lowerCaseQuery)) {
                    filteredIngredients.add(ingredient);
                }
            }
        }

        notifyDataSetChanged();
    }

    /**
     * Obtiene ingredientes seleccionados
     */
    public List<Ingredient> getSelectedIngredients() {
        List<Ingredient> selectedIngredients = new ArrayList<>();
        for (Ingredient ingredient : ingredients) {
            if (ingredient.isSelected()) {
                selectedIngredients.add(ingredient);
            }
        }
        return selectedIngredients;
    }

    /**
     * Selecciona/deselecciona todos los ingredientes
     */
    public void selectAll(boolean select) {
        for (Ingredient ingredient : ingredients) {
            ingredient.setSelected(select);
        }
        notifyDataSetChanged();
    }

    /**
     * Obtiene el número de ingredientes seleccionados
     */
    public int getSelectedCount() {
        int count = 0;
        for (Ingredient ingredient : ingredients) {
            if (ingredient.isSelected()) {
                count++;
            }
        }
        return count;
    }

    static class IngredientViewHolder extends RecyclerView.ViewHolder {
        private CheckBox selectionCheckBox;
        private TextView nameText;
        private TextView costText;
        private TextView costPerTonneText;
        private TextView nutrientsText;
        private View infoButton;
        private View itemContainer;

        public IngredientViewHolder(@NonNull View itemView) {
            super(itemView);

            selectionCheckBox = itemView.findViewById(R.id.ingredientCheckBox);
            nameText = itemView.findViewById(R.id.ingredientNameText);
            costText = itemView.findViewById(R.id.ingredientCostText);
            costPerTonneText = itemView.findViewById(R.id.ingredientCostPerTonneText);
            nutrientsText = itemView.findViewById(R.id.ingredientNutrientsText);
            infoButton = itemView.findViewById(R.id.ingredientInfoButton);
            itemContainer = itemView.findViewById(R.id.ingredientItemContainer);
        }

        public void bind(Ingredient ingredient, OnIngredientSelectionListener listener) {
            // Configurar nombre
            nameText.setText(ingredient.getName());

            // Configurar costos
            costText.setText(String.format("$%.3f/kg", ingredient.getCost()));
            costPerTonneText.setText(String.format("$%.0f/ton", ingredient.getCostPerTonne()));

            // Configurar información nutricional resumida
            StringBuilder nutrientsInfo = new StringBuilder();
            if (ingredient.getNutrients().containsKey("CP")) {
                nutrientsInfo.append(String.format("CP: %.1f%% ", ingredient.getNutrient("CP")));
            }
            if (ingredient.getNutrients().containsKey("TDN")) {
                nutrientsInfo.append(String.format("TDN: %.1f%% ", ingredient.getNutrient("TDN")));
            }
            if (ingredient.getNutrients().containsKey("NDF")) {
                nutrientsInfo.append(String.format("NDF: %.1f%%", ingredient.getNutrient("NDF")));
            }

            nutrientsText.setText(nutrientsInfo.toString().trim());

            // Configurar checkbox
            selectionCheckBox.setOnCheckedChangeListener(null); // Remover listener temporal
            selectionCheckBox.setChecked(ingredient.isSelected());

            // Configurar listeners
            selectionCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
                ingredient.setSelected(isChecked);
                updateItemAppearance(ingredient.isSelected());
                if (listener != null) {
                    listener.onIngredientSelected(ingredient, isChecked);
                }
            });

            // Listener para el container completo (también cambia el checkbox)
            itemContainer.setOnClickListener(v -> {
                boolean newState = !ingredient.isSelected();
                ingredient.setSelected(newState);
                selectionCheckBox.setChecked(newState);
                updateItemAppearance(newState);
                if (listener != null) {
                    listener.onIngredientSelected(ingredient, newState);
                }
            });

            // Listener para el botón de información
            infoButton.setOnClickListener(v -> {
                if (listener != null) {
                    listener.onIngredientInfoClicked(ingredient);
                }
            });

            // Actualizar apariencia inicial
            updateItemAppearance(ingredient.isSelected());
        }

        private void updateItemAppearance(boolean isSelected) {
            if (isSelected) {
                itemContainer.setBackgroundResource(R.drawable.item_ingredient_selected_background);
                nameText.setTextColor(itemView.getContext().getResources().getColor(R.color.green_dark));
            } else {
                itemContainer.setBackgroundResource(R.drawable.item_ingredient_background);
                nameText.setTextColor(itemView.getContext().getResources().getColor(android.R.color.black));
            }
        }
    }
}
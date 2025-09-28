package com.example.dietoptimizerapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.dietoptimizerapp.models.AnimalType;
import com.example.dietoptimizerapp.models.DietResult;
import com.example.dietoptimizerapp.models.Ingredient;
import com.example.dietoptimizerapp.adapters.ResultsAdapter;
import com.example.dietoptimizerapp.utils.CostCalculator;
import com.example.dietoptimizerapp.utils.JsonBuilder;
import com.example.dietoptimizerapp.jni.JniBridge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "DietOptimizer";

    // UI Components
    private Spinner animalSpinner, ingredientSpinner, dietTypeSpinner;
    private EditText quantityEditText;
    private Button addIngredientBtn, calculateBtn;
    private LinearLayout selectedIngredientsLayout;
    private ProgressBar progressBar;
    private RecyclerView resultsRecyclerView;
    private TextView statusTextView;

    // Data
    private List<Ingredient> availableIngredients;
    private List<Ingredient> selectedIngredients;
    private AnimalType[] animalTypes;
    private String[] dietTypes;

    // Adapters
    private ResultsAdapter resultsAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeUI();
        initializeData();
        setupSpinners();
        setupListeners();
    }

    private void initializeUI() {
        animalSpinner = findViewById(R.id.animalSpinner);
        ingredientSpinner = findViewById(R.id.ingredientSpinner);
        dietTypeSpinner = findViewById(R.id.dietTypeSpinner);
        quantityEditText = findViewById(R.id.quantityEditText);
        addIngredientBtn = findViewById(R.id.addIngredientBtn);
        calculateBtn = findViewById(R.id.calculateBtn);
        selectedIngredientsLayout = findViewById(R.id.selectedIngredientsLayout);
        progressBar = findViewById(R.id.progressBar);
        resultsRecyclerView = findViewById(R.id.resultsRecyclerView);
        statusTextView = findViewById(R.id.statusTextView);

        // Setup RecyclerView
        resultsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        resultsAdapter = new ResultsAdapter(new ArrayList<>());
        resultsRecyclerView.setAdapter(resultsAdapter);
    }

    private void initializeData() {
        // Initialize available ingredients with your JSON data
        availableIngredients = createAvailableIngredients();
        selectedIngredients = new ArrayList<>();

        // Initialize animal types
        animalTypes = AnimalType.getDefaultAnimalTypes();

        // Initialize diet types
        dietTypes = new String[]{"Mantenimiento", "Crecimiento", "Lactación", "Engorde"};
    }

    private List<Ingredient> createAvailableIngredients() {
        List<Ingredient> ingredients = new ArrayList<>();

        // Maiz Grano -> MaizGrano
        Map<String, Double> maizNutrients = new HashMap<>();
        maizNutrients.put("GE", 4.44); // Mcal/kg MS
        maizNutrients.put("CP", 9.0); // %
        maizNutrients.put("TDN", 90.0); // %
        maizNutrients.put("NEm", 2.21); // Mcal/kg MS
        maizNutrients.put("Ca", 0.02); // %
        maizNutrients.put("P", 0.31); // %
        maizNutrients.put("NDF", 10.5); // %
        ingredients.add(new Ingredient("MaizGrano", 0.22, maizNutrients));

// Harina de Soja -> HarinaSoja (48.5% PB)
        Map<String, Double> sojaNutrients = new HashMap<>();
        sojaNutrients.put("GE", 4.71); // Mcal/kg MS
        sojaNutrients.put("CP", 50.0); // % (Ajustado a 48.5% PB en base húmeda, que es ~50% en MS)
        sojaNutrients.put("TDN", 85.0); // %
        sojaNutrients.put("NEm", 2.08); // Mcal/kg MS
        sojaNutrients.put("Ca", 0.38); // %
        sojaNutrients.put("P", 0.76); // %
        sojaNutrients.put("NDF", 7.5); // %
        ingredients.add(new Ingredient("HarinaSoja", 0.48, sojaNutrients));

// Heno de Alfalfa -> HenoAlfalfa (Calidad media)
        Map<String, Double> alfalfaNutrients = new HashMap<>();
        alfalfaNutrients.put("GE", 4.20); // Mcal/kg MS
        alfalfaNutrients.put("CP", 18.0); // %
        alfalfaNutrients.put("TDN", 62.0); // %
        alfalfaNutrients.put("NEm", 1.35); // Mcal/kg MS
        alfalfaNutrients.put("Ca", 1.45); // %
        alfalfaNutrients.put("P", 0.26); // %
        alfalfaNutrients.put("NDF", 44.0); // %
        ingredients.add(new Ingredient("HenoAlfalfa", 0.15, alfalfaNutrients));

// Granos Destileria (DDGS) -> DDGS (Maíz)
        Map<String, Double> ddgsNutrients = new HashMap<>();
        ddgsNutrients.put("GE", 5.30); // Mcal/kg MS - Valor más alto por la concentración de grasa.
        ddgsNutrients.put("CP", 31.0); // %
        ddgsNutrients.put("TDN", 89.0); // %
        ddgsNutrients.put("NEm", 2.15); // Mcal/kg MS
        ddgsNutrients.put("Ca", 0.07); // % - Valor real es bastante bajo.
        ddgsNutrients.put("P", 0.82); // % - Fuente rica en fósforo.
        ddgsNutrients.put("NDF", 38.0); // %
        ingredients.add(new Ingredient("DDGS", 0.25, ddgsNutrients));

// Premix Mineral-Vit -> PremixMineral
// Los valores pueden variar enormemente. Estos representan un premix común para bovinos.
        Map<String, Double> premixNutrients = new HashMap<>();
        premixNutrients.put("GE", 0.0);
        premixNutrients.put("CP", 0.0);
        premixNutrients.put("TDN", 0.0);
        premixNutrients.put("NEm", 0.0);
        premixNutrients.put("Ca", 24.0); // % - Común en premezclas altas en calcio.
        premixNutrients.put("P", 12.0); // % - Ratio Ca:P de 2:1 es típico.
        premixNutrients.put("NDF", 0.0);
        ingredients.add(new Ingredient("PremixMineral", 1.50, premixNutrients));

// Melaza de Caña -> MelazaCana
        Map<String, Double> melazaNutrients = new HashMap<>();
        melazaNutrients.put("GE", 4.10); // Mcal/kg MS
        melazaNutrients.put("CP", 7.5); // % - Puede variar, pero 4% es bajo.
        melazaNutrients.put("TDN", 78.0); // %
        melazaNutrients.put("NEm", 1.85); // Mcal/kg MS
        melazaNutrients.put("Ca", 1.0); // %
        melazaNutrients.put("P", 0.08); // %
        melazaNutrients.put("NDF", 0.0); // % - No contiene fibra estructural.
        ingredients.add(new Ingredient("MelazaCana", 0.18, melazaNutrients));

// Salvado de Trigo -> SalvadoTrigo
        Map<String, Double> salvadoNutrients = new HashMap<>();
        salvadoNutrients.put("GE", 4.52); // Mcal/kg MS
        salvadoNutrients.put("CP", 17.0); // %
        salvadoNutrients.put("TDN", 70.0); // %
        salvadoNutrients.put("NEm", 1.62); // Mcal/kg MS
        salvadoNutrients.put("Ca", 0.13); // %
        salvadoNutrients.put("P", 1.18); // % - Muy alto en fósforo, un punto clave de este ingrediente.
        salvadoNutrients.put("NDF", 45.0); // %
        ingredients.add(new Ingredient("SalvadoTrigo", 0.20, salvadoNutrients));

        return ingredients;
    }

    private void setupSpinners() {
        // Animal spinner
        ArrayAdapter<AnimalType> animalAdapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, animalTypes);
        animalAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        animalSpinner.setAdapter(animalAdapter);

        // Ingredient spinner
        updateIngredientSpinner();

        // Diet type spinner
        ArrayAdapter<String> dietAdapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, dietTypes);
        dietAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        dietTypeSpinner.setAdapter(dietAdapter);
    }

    private void updateIngredientSpinner() {
        List<Ingredient> availableForSelection = new ArrayList<>();
        for (Ingredient ingredient : availableIngredients) {
            if (!ingredient.isSelected()) {
                availableForSelection.add(ingredient);
            }
        }

        ArrayAdapter<Ingredient> ingredientAdapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, availableForSelection);
        ingredientAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        ingredientSpinner.setAdapter(ingredientAdapter);
    }

    private void setupListeners() {
        addIngredientBtn.setOnClickListener(v -> addSelectedIngredient());
        calculateBtn.setOnClickListener(v -> calculateDiets());
    }

    private void addSelectedIngredient() {
        if (ingredientSpinner.getSelectedItem() == null) {
            Toast.makeText(this, "Selecciona un ingrediente", Toast.LENGTH_SHORT).show();
            return;
        }

        Ingredient selectedIngredient = (Ingredient) ingredientSpinner.getSelectedItem();
        selectedIngredient.setSelected(true);
        selectedIngredients.add(selectedIngredient);

        // Add visual representation
        addSelectedIngredientView(selectedIngredient);

        // Update spinner
        updateIngredientSpinner();

        // Enable calculate button if we have at least one ingredient and animal
        updateCalculateButton();
    }

    private void addSelectedIngredientView(Ingredient ingredient) {
        View ingredientView = getLayoutInflater().inflate(R.layout.item_selected_ingredient, null);

        TextView nameText = ingredientView.findViewById(R.id.ingredientNameText);
        TextView costText = ingredientView.findViewById(R.id.ingredientCostText);
        Button removeBtn = ingredientView.findViewById(R.id.removeIngredientBtn);

        nameText.setText(ingredient.getName());
        costText.setText(String.format("$%.3f/kg ($%.0f/ton)", ingredient.getCost(), ingredient.getCostPerTonne()));

        removeBtn.setOnClickListener(v -> {
            removeSelectedIngredient(ingredient, ingredientView);
        });

        selectedIngredientsLayout.addView(ingredientView);
    }

    private void removeSelectedIngredient(Ingredient ingredient, View ingredientView) {
        ingredient.setSelected(false);
        selectedIngredients.remove(ingredient);
        selectedIngredientsLayout.removeView(ingredientView);
        updateIngredientSpinner();
        updateCalculateButton();
    }

    private void updateCalculateButton() {
        boolean hasIngredients = !selectedIngredients.isEmpty();
        boolean hasAnimal = animalSpinner.getSelectedItem() != null;
        calculateBtn.setEnabled(hasIngredients && hasAnimal);
    }

    private void calculateDiets() {
        if (animalSpinner.getSelectedItem() == null || selectedIngredients.isEmpty()) {
            Toast.makeText(this, "Selecciona un animal y al menos un ingrediente", Toast.LENGTH_SHORT).show();
            return;
        }
        if (selectedIngredients.isEmpty()) {
            Toast.makeText(this, "No hay ingredientes seleccionados", Toast.LENGTH_SHORT).show();
            return;
        }
        Log.i(TAG, "Ingredientes seleccionados: " + selectedIngredients.size());
        for (Ingredient ing : selectedIngredients) {
            Log.i(TAG, "- " + ing.getName() + " (selected: " + ing.isSelected() + ")");
        }

        // Show progress
        progressBar.setVisibility(View.VISIBLE);
        calculateBtn.setEnabled(false);
        statusTextView.setText("Calculando dietas óptimas...");

        // Get selected values
        AnimalType selectedAnimal = (AnimalType) animalSpinner.getSelectedItem();
        String selectedDietType = (String) dietTypeSpinner.getSelectedItem();
        int quantity = getQuantity();

        for (Ingredient ingredient : selectedIngredients) {
            ingredient.setSelected(true);
        }

        // Build JSON
        String jsonInput = JsonBuilder.buildDietRequestJson(selectedAnimal, selectedIngredients, selectedDietType);

        Log.i(TAG, "JSON completo length: " + jsonInput.length());
        Log.i(TAG, "JSON Input: " + jsonInput);

        // Call native library in background thread
        new Thread(() -> {
            try {
                ArrayList<DietResult> results = JniBridge.calculateDiets(jsonInput);

                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    calculateBtn.setEnabled(true);

                    if (results != null && !results.isEmpty()) {
                        statusTextView.setText("Se encontraron " + results.size() + " dietas óptimas");
                        displayResults(results, selectedAnimal, quantity);
                    } else {
                        statusTextView.setText("No se encontraron dietas factibles");
                        resultsAdapter.updateResults(new ArrayList<>());
                        Toast.makeText(MainActivity.this, "No se pudo encontrar una dieta que cumpla todos los requerimientos", Toast.LENGTH_LONG).show();
                    }
                });

            } catch (Exception e) {
                Log.e(TAG, "Error calculando dietas", e);
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    calculateBtn.setEnabled(true);
                    statusTextView.setText("Error en el cálculo");
                    Toast.makeText(MainActivity.this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private void displayResults(ArrayList<DietResult> results, AnimalType animal, int quantity) {
        // Prepare results with cost calculations
        List<ResultsAdapter.DietResultWithCosts> resultsWithCosts = new ArrayList<>();

        for (int i = 0; i < results.size(); i++) {
            DietResult diet = results.get(i);
            CostCalculator.CostBreakdown costs = CostCalculator.calculateCosts(diet, animal.getDmiKgDay(), quantity);

            String title = "Opción " + (i + 1);
            if (i == 0) title += " - Mejor Costo";
            else if (i == 1) title += " - Balance";
            else title += " - Menos Emisiones";

            resultsWithCosts.add(new ResultsAdapter.DietResultWithCosts(diet, costs, title));
        }

        resultsAdapter.updateResults(resultsWithCosts);
    }

    private int getQuantity() {
        try {
            String quantityStr = quantityEditText.getText().toString().trim();
            if (quantityStr.isEmpty()) return 1;
            int quantity = Integer.parseInt(quantityStr);
            return Math.max(1, quantity);
        } catch (NumberFormatException e) {
            return 1;
        }
    }
}
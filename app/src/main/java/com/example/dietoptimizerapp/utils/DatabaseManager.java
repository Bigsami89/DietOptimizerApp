package com.example.dietoptimizerapp.utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.net.Uri;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.dietoptimizerapp.models.AnimalType;
import com.example.dietoptimizerapp.models.Ingredient;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

/**
 * Gestor centralizado para la base de datos JSON localizada en assets/data.
 * La clase aplica caché en memoria y provee utilidades básicas de búsqueda y actualización.
 */
public class DatabaseManager {

    private static final String TAG = "DatabaseManager";
    private static final String INGREDIENTS_PATH = "data/ingredients_database.json";
    private static final String ANIMALS_PATH = "data/animal_types_database.json";

    private static volatile DatabaseManager instance;

    private final Context appContext;
    private final Gson gson;

    private List<Ingredient> ingredientsCache;
    private List<AnimalType> animalTypesCache;

    private DatabaseMetadata ingredientMetadata;
    private DatabaseMetadata animalMetadata;

    private DatabaseManager(@NonNull Context context) {
        this.appContext = context.getApplicationContext();
        this.gson = new Gson();
        this.ingredientsCache = null;
        this.animalTypesCache = null;
    }

    public static DatabaseManager getInstance(@NonNull Context context) {
        if (instance == null) {
            synchronized (DatabaseManager.class) {
                if (instance == null) {
                    instance = new DatabaseManager(context);
                }
            }
        }
        return instance;
    }

    public synchronized List<Ingredient> getAllIngredients() {
        ensureIngredientsLoaded();
        return ingredientsCache == null ? Collections.emptyList() : new ArrayList<>(ingredientsCache);
    }

    public synchronized List<AnimalType> getAllAnimalTypes() {
        ensureAnimalsLoaded();
        return animalTypesCache == null ? Collections.emptyList() : new ArrayList<>(animalTypesCache);
    }

    @Nullable
    public synchronized Ingredient getIngredientById(@NonNull String id) {
        ensureIngredientsLoaded();
        if (ingredientsCache == null) {
            return null;
        }
        for (Ingredient ingredient : ingredientsCache) {
            if (id.equalsIgnoreCase(ingredient.getId())) {
                return ingredient;
            }
        }
        return null;
    }

    @Nullable
    public synchronized AnimalType getAnimalTypeById(@NonNull String id) {
        ensureAnimalsLoaded();
        if (animalTypesCache == null) {
            return null;
        }
        for (AnimalType animal : animalTypesCache) {
            if (id.equalsIgnoreCase(animal.getId())) {
                return animal;
            }
        }
        return null;
    }

    public synchronized List<Ingredient> searchIngredients(@NonNull String query) {
        ensureIngredientsLoaded();
        String normalized = query.trim().toLowerCase(Locale.getDefault());
        if (normalized.isEmpty() || ingredientsCache == null) {
            return getAllIngredients();
        }
        List<Ingredient> matches = new ArrayList<>();
        for (Ingredient ingredient : ingredientsCache) {
            if (ingredient.getName().toLowerCase(Locale.getDefault()).contains(normalized)
                    || ingredient.getLocalName().toLowerCase(Locale.getDefault()).contains(normalized)
                    || ingredient.getCategory().toLowerCase(Locale.getDefault()).contains(normalized)) {
                matches.add(ingredient);
            }
        }
        return matches;
    }

    public synchronized List<Ingredient> filterByCategory(@NonNull String category) {
        ensureIngredientsLoaded();
        if (ingredientsCache == null) {
            return Collections.emptyList();
        }
        List<Ingredient> filtered = new ArrayList<>();
        String normalized = category.toLowerCase(Locale.getDefault());
        for (Ingredient ingredient : ingredientsCache) {
            if (ingredient.getCategory().toLowerCase(Locale.getDefault()).equals(normalized)) {
                filtered.add(ingredient);
            }
        }
        return filtered;
    }

    public synchronized void updateIngredientCost(@NonNull String id, double newCost) {
        Ingredient ingredient = getIngredientById(id);
        if (ingredient != null) {
            ingredient.setCost(newCost);
            ingredient.setCustom(true);
        }
    }

    public synchronized void addCustomIngredient(@NonNull Ingredient ingredient) {
        ensureIngredientsLoaded();
        if (ingredientsCache == null) {
            ingredientsCache = new ArrayList<>();
        }
        if (ingredient.getId() == null || ingredient.getId().isEmpty()) {
            ingredient.setId("custom-" + UUID.randomUUID());
        }
        ingredient.setCustom(true);
        ingredientsCache.add(ingredient);
    }

    public synchronized boolean exportDatabase() {
        ensureIngredientsLoaded();
        ensureAnimalsLoaded();

        File exportDir = appContext.getExternalFilesDir(null);
        if (exportDir == null) {
            Log.e(TAG, "No external storage directory available for export");
            return false;
        }

        boolean success = true;
        success &= writeJsonToFile(new File(exportDir, "ingredients_database_export.json"), buildIngredientExportJson());
        success &= writeJsonToFile(new File(exportDir, "animal_types_database_export.json"), buildAnimalExportJson());
        return success;
    }

    public synchronized boolean importDatabase(@NonNull File jsonFile) {
        if (!jsonFile.exists()) {
            Log.w(TAG, "Archivo JSON no encontrado: " + jsonFile.getAbsolutePath());
            return false;
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(jsonFile))) {
            JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
            if (root.has("ingredients")) {
                parseIngredients(root, false);
            } else if (root.has("animal_types")) {
                parseAnimals(root, false);
            } else {
                Log.w(TAG, "Estructura JSON desconocida en importación");
                return false;
            }
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error leyendo archivo de importación", e);
            return false;
        }
    }

    public synchronized boolean validateDatabase() {
        ensureIngredientsLoaded();
        ensureAnimalsLoaded();
        return ingredientsCache != null && ingredientsCache.size() >= 20
                && animalTypesCache != null && animalTypesCache.size() >= 8;
    }

    @Nullable
    public Uri getIngredientDatabaseUriForExport() {
        File exportDir = appContext.getExternalFilesDir(null);
        if (exportDir == null) {
            return null;
        }
        File file = new File(exportDir, "ingredients_database_export.json");
        return Uri.fromFile(file);
    }

    private void ensureIngredientsLoaded() {
        if (ingredientsCache != null) {
            return;
        }
        try {
            AssetManager assets = appContext.getAssets();
            try (InputStream inputStream = assets.open(INGREDIENTS_PATH);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
                JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
                parseIngredients(root, true);
            }
        } catch (FileNotFoundException e) {
            Log.e(TAG, "Archivo de ingredientes no encontrado", e);
        } catch (IOException e) {
            Log.e(TAG, "Error leyendo ingredientes", e);
        }
    }

    private void ensureAnimalsLoaded() {
        if (animalTypesCache != null) {
            return;
        }
        try {
            AssetManager assets = appContext.getAssets();
            try (InputStream inputStream = assets.open(ANIMALS_PATH);
                 BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
                JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
                parseAnimals(root, true);
            }
        } catch (FileNotFoundException e) {
            Log.e(TAG, "Archivo de tipos de animales no encontrado", e);
        } catch (IOException e) {
            Log.e(TAG, "Error leyendo tipos de animales", e);
        }
    }

    private void parseIngredients(@NonNull JsonObject root, boolean fromAssets) {
        ingredientMetadata = extractMetadata(root);
        JsonArray ingredients = root.getAsJsonArray("ingredients");
        List<Ingredient> parsed = new ArrayList<>();
        for (JsonElement element : ingredients) {
            JsonObject obj = element.getAsJsonObject();
            Ingredient ingredient = new Ingredient(
                    obj.get("id").getAsString(),
                    obj.get("name").getAsString(),
                    obj.get("cost").getAsDouble()
            );
            ingredient.setLocalName(obj.get("local_name").getAsString());
            ingredient.setCategory(obj.get("category").getAsString());
            ingredient.setCostUnit(obj.get("cost_unit").getAsString());
            if (obj.has("is_forage")) {
                ingredient.setForage(obj.get("is_forage").getAsBoolean());
            }
            ingredient.setAvailability(obj.get("availability").getAsString());
            ingredient.setStorageRequirements(obj.get("storage_requirements").getAsString());
            ingredient.setPalatabilityScore(obj.get("palatability_score").getAsInt());
            ingredient.setMinInclusion(obj.get("min_inclusion").getAsDouble());
            ingredient.setMaxInclusion(obj.get("max_inclusion").getAsDouble());
            ingredient.setRecommendedInclusion(obj.get("recommended_inclusion").getAsDouble());
            ingredient.setDescription(obj.get("description").getAsString());
            ingredient.setNotes(obj.has("notes") ? obj.get("notes").getAsString() : "");

            Map<String, Double> nutrients = new HashMap<>();
            JsonObject nutrientsObj = obj.getAsJsonObject("nutrients");
            for (Map.Entry<String, JsonElement> entry : nutrientsObj.entrySet()) {
                nutrients.put(entry.getKey(), entry.getValue().getAsDouble());
            }
            ingredient.setNutrients(nutrients);

            Map<String, Double> variability = new HashMap<>();
            if (obj.has("nutrient_variability")) {
                JsonObject variabilityObj = obj.getAsJsonObject("nutrient_variability");
                for (Map.Entry<String, JsonElement> entry : variabilityObj.entrySet()) {
                    variability.put(entry.getKey(), entry.getValue().getAsDouble());
                }
            }
            ingredient.setNutrientVariability(variability);

            ingredient.setAntiNutritionalFactors(asStringList(obj.getAsJsonArray("anti_nutritional_factors")));
            ingredient.setMixingRestrictions(asStringList(obj.getAsJsonArray("mixing_restrictions")));
            ingredient.setImages(asStringList(obj.getAsJsonArray("images")));
            ingredient.setReferences(asStringList(obj.getAsJsonArray("references")));

            if (!fromAssets && obj.has("is_custom") && obj.get("is_custom").getAsBoolean()) {
                ingredient.setCustom(true);
            }

            parsed.add(ingredient);
        }
        ingredientsCache = parsed;
    }

    private void parseAnimals(@NonNull JsonObject root, boolean fromAssets) {
        animalMetadata = extractMetadata(root);
        JsonArray animals = root.getAsJsonArray("animal_types");
        List<AnimalType> parsed = new ArrayList<>();
        for (JsonElement element : animals) {
            JsonObject obj = element.getAsJsonObject();
            AnimalType animal = new AnimalType(
                    obj.get("id").getAsString(),
                    obj.get("name").getAsString(),
                    obj.get("dmi_kg_day").getAsDouble(),
                    obj.get("body_weight_kg").getAsDouble()
            );
            animal.setSpecies(obj.get("species").getAsString());
            animal.setBreed(obj.get("breed").getAsString());
            animal.setProductionType(obj.get("production_type").getAsString());
            animal.setLifeStage(obj.get("life_stage").getAsString());
            animal.setBodyWeightRange(parseRange(obj.getAsJsonArray("body_weight_range")));
            animal.setDmiFormula(obj.get("dmi_formula").getAsString());
            animal.setDescription(obj.get("description").getAsString());
            animal.setImage(obj.get("image").getAsString());
            animal.setLastUpdated(obj.get("last_updated").getAsString());

            Map<String, Double> minReqs = mapFromJson(obj.getAsJsonObject("min_requirements"));
            Map<String, Double> maxReqs = mapFromJson(obj.getAsJsonObject("max_requirements"));
            animal.setMinRequirements(minReqs);
            animal.setMaxRequirements(maxReqs);

            JsonObject optimalRangesObj = obj.getAsJsonObject("optimal_ranges");
            if (optimalRangesObj != null) {
                Map<String, AnimalType.RequirementRange> optimalRanges = new HashMap<>();
                for (Map.Entry<String, JsonElement> entry : optimalRangesObj.entrySet()) {
                    JsonArray range = entry.getValue().getAsJsonArray();
                    optimalRanges.put(entry.getKey(), new AnimalType.RequirementRange(
                            range.get(0).getAsDouble(),
                            range.get(1).getAsDouble()
                    ));
                }
                animal.setOptimalRanges(optimalRanges);
            }

            animal.setDietAdjustments(mapFromJson(obj.getAsJsonObject("diet_adjustments")));
            animal.setProductivityTargets(mapFromJson(obj.getAsJsonObject("productivity_targets")));
            animal.setEnvironmentalFactors(mapFromJson(obj.getAsJsonObject("environmental_factors")));
            animal.setHealthConsiderations(asStringList(obj.getAsJsonArray("health_considerations")));

            parsed.add(animal);
        }
        animalTypesCache = parsed;
    }

    private double[] parseRange(@Nullable JsonArray array) {
        if (array == null || array.size() < 2) {
            return new double[]{0, 0};
        }
        return new double[]{array.get(0).getAsDouble(), array.get(1).getAsDouble()};
    }

    private Map<String, Double> mapFromJson(@Nullable JsonObject jsonObject) {
        Map<String, Double> map = new HashMap<>();
        if (jsonObject == null) {
            return map;
        }
        for (Map.Entry<String, JsonElement> entry : jsonObject.entrySet()) {
            map.put(entry.getKey(), entry.getValue().getAsDouble());
        }
        return map;
    }

    private List<String> asStringList(@Nullable JsonArray jsonArray) {
        List<String> list = new ArrayList<>();
        if (jsonArray == null) {
            return list;
        }
        for (JsonElement element : jsonArray) {
            list.add(element.getAsString());
        }
        return list;
    }

    private boolean writeJsonToFile(@NonNull File file, @NonNull JsonObject json) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            gson.toJson(json, writer);
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Error exportando JSON", e);
            return false;
        }
    }

    private JsonObject buildIngredientExportJson() {
        JsonObject root = new JsonObject();
        if (ingredientMetadata != null) {
            ingredientMetadata.applyTo(root);
        }
        JsonArray array = new JsonArray();
        if (ingredientsCache != null) {
            for (Ingredient ingredient : ingredientsCache) {
                JsonObject obj = new JsonObject();
                obj.addProperty("id", ingredient.getId());
                obj.addProperty("name", ingredient.getName());
                obj.addProperty("local_name", ingredient.getLocalName());
                obj.addProperty("category", ingredient.getCategory());
                obj.addProperty("cost", ingredient.getCost());
                obj.addProperty("cost_unit", ingredient.getCostUnit());
                obj.addProperty("availability", ingredient.getAvailability());
                obj.addProperty("storage_requirements", ingredient.getStorageRequirements());
                obj.addProperty("palatability_score", ingredient.getPalatabilityScore());
                obj.addProperty("min_inclusion", ingredient.getMinInclusion());
                obj.addProperty("max_inclusion", ingredient.getMaxInclusion());
                obj.addProperty("recommended_inclusion", ingredient.getRecommendedInclusion());
                obj.addProperty("description", ingredient.getDescription());
                obj.addProperty("notes", ingredient.getNotes());
                obj.addProperty("is_custom", ingredient.isCustom());

                JsonObject nutrients = new JsonObject();
                for (Map.Entry<String, Double> entry : ingredient.getNutrients().entrySet()) {
                    nutrients.addProperty(entry.getKey(), entry.getValue());
                }
                obj.add("nutrients", nutrients);

                JsonObject variability = new JsonObject();
                for (Map.Entry<String, Double> entry : ingredient.getNutrientVariability().entrySet()) {
                    variability.addProperty(entry.getKey(), entry.getValue());
                }
                obj.add("nutrient_variability", variability);

                obj.add("anti_nutritional_factors", gson.toJsonTree(new ArrayList<>(ingredient.getAntiNutritionalFactors())));
                obj.add("mixing_restrictions", gson.toJsonTree(new ArrayList<>(ingredient.getMixingRestrictions())));
                obj.add("images", gson.toJsonTree(new ArrayList<>(ingredient.getImages())));
                obj.add("references", gson.toJsonTree(new ArrayList<>(ingredient.getReferences())));
                array.add(obj);
            }
        }
        root.add("ingredients", array);
        return root;
    }

    private JsonObject buildAnimalExportJson() {
        JsonObject root = new JsonObject();
        if (animalMetadata != null) {
            animalMetadata.applyTo(root);
        }
        JsonArray array = new JsonArray();
        if (animalTypesCache != null) {
            for (AnimalType animal : animalTypesCache) {
                JsonObject obj = new JsonObject();
                obj.addProperty("id", animal.getId());
                obj.addProperty("name", animal.getName());
                obj.addProperty("species", animal.getSpecies());
                obj.addProperty("breed", animal.getBreed());
                obj.addProperty("production_type", animal.getProductionType());
                obj.addProperty("life_stage", animal.getLifeStage());
                obj.addProperty("dmi_kg_day", animal.getDmiKgDay());
                obj.addProperty("body_weight_kg", animal.getBodyWeightKg());
                obj.addProperty("dmi_formula", animal.getDmiFormula());
                obj.addProperty("description", animal.getDescription());
                obj.addProperty("image", animal.getImage());
                obj.addProperty("last_updated", animal.getLastUpdated());

                JsonArray range = new JsonArray();
                double[] bodyWeightRange = animal.getBodyWeightRange();
                range.add(bodyWeightRange[0]);
                range.add(bodyWeightRange[1]);
                obj.add("body_weight_range", range);

                obj.add("min_requirements", gson.toJsonTree(animal.getMinRequirements()));
                obj.add("max_requirements", gson.toJsonTree(animal.getMaxRequirements()));

                JsonObject optimalRanges = new JsonObject();
                for (Map.Entry<String, AnimalType.RequirementRange> entry : animal.getOptimalRanges().entrySet()) {
                    JsonArray value = new JsonArray();
                    value.add(entry.getValue().getMin());
                    value.add(entry.getValue().getMax());
                    optimalRanges.add(entry.getKey(), value);
                }
                obj.add("optimal_ranges", optimalRanges);

                obj.add("diet_adjustments", gson.toJsonTree(animal.getDietAdjustments()));
                obj.add("productivity_targets", gson.toJsonTree(animal.getProductivityTargets()));
                obj.add("environmental_factors", gson.toJsonTree(animal.getEnvironmentalFactors()));
                obj.add("health_considerations", gson.toJsonTree(new ArrayList<>(animal.getHealthConsiderations())));

                array.add(obj);
            }
        }
        root.add("animal_types", array);
        return root;
    }

    private DatabaseMetadata extractMetadata(@NonNull JsonObject root) {
        DatabaseMetadata metadata = new DatabaseMetadata();
        boolean hasData = false;
        if (root.has("version")) {
            metadata.version = root.get("version").getAsString();
            hasData = true;
        }
        if (root.has("last_updated")) {
            metadata.lastUpdated = root.get("last_updated").getAsString();
            hasData = true;
        }
        if (root.has("metadata")) {
            metadata.additional = root.getAsJsonObject("metadata");
            hasData = true;
        }
        return hasData ? metadata : null;
    }

    private static class DatabaseMetadata {
        private String version = "";
        private String lastUpdated = "";
        private JsonObject additional;

        void applyTo(@NonNull JsonObject root) {
            if (version != null && !version.isEmpty()) {
                root.addProperty("version", version);
            }
            if (lastUpdated != null && !lastUpdated.isEmpty()) {
                root.addProperty("last_updated", lastUpdated);
            }
            if (additional != null) {
                root.add("metadata", additional.deepCopy());
            }
        }
    }
}

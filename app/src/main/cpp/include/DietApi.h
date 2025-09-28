#pragma once

#if defined(_WIN32)
    #ifdef API_EXPORT_DEFINITION
        #define API_FUNCTION __declspec(dllexport)
    #else
        #define API_FUNCTION __declspec(dllimport)
    #endif
#else
    #define API_FUNCTION
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Estructuras de Datos para la API en C
typedef struct {
    const char* name;
    double cost;
    double minInclusion;
    double maxInclusion;
    const char** propertyKeys;
    double* propertyValues;
    int propertiesCount;
} C_Ingredient;

typedef struct {
    const char* nutrientName;
    double requiredValue;
} C_NutrientRequirement;

typedef struct {
    const char* ingredientName;
    double proportion;
} C_DietComponent;

typedef struct {
    C_DietComponent* components;
    int componentsCount;
    double totalCost;
    double totalMethane;
} C_DietResult;

API_FUNCTION int calculate_diets_from_json(
    const char* json_string,
    C_DietResult** out_results,
    int* out_results_count
);

API_FUNCTION void free_diet_results(C_DietResult* results, int results_count);

#ifdef __cplusplus
}
#endif
#include "DietApi.h"
#include "RationCalculator.h"
#include "SimplexSolver.h"
#include "ApiRequest.h"
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

API_FUNCTION int calculate_diets_from_json(
    const char* json_string,
    C_DietResult** out_results,
    int* out_results_count) {
    
    if (!json_string || !out_results || !out_results_count) return -1;

    ApiRequest request;
    if (!ApiRequest::fromJson(json_string, request)) return -2; // Error de parseo

    try {
        auto solver = std::make_unique<SimplexSolver>();
        RationCalculator calculator(std::move(solver));
        std::vector<Diet> solutions = calculator.findTopDiets(request);
        
        *out_results_count = static_cast<int>(solutions.size());
        *out_results = new C_DietResult[*out_results_count];

        for (int i = 0; i < *out_results_count; ++i) {
            const auto& sol = solutions[i];
            (*out_results)[i].componentsCount = static_cast<int>(sol.composition.size());
            (*out_results)[i].components = new C_DietComponent[sol.composition.size()];
            (*out_results)[i].totalCost = sol.totalCost;
            (*out_results)[i].totalMethane = sol.entericMethane;

            for (size_t j = 0; j < sol.composition.size(); ++j) {
                const auto& comp = sol.composition[j];
                size_t name_len = comp.first.name.length() + 1;
                char* name_copy = new char[name_len];
                
                // Usar strcpy_s para mayor seguridad en Windows
                #if defined(_WIN32)
                    strcpy_s(name_copy, name_len, comp.first.name.c_str());
                #else
                    strcpy(name_copy, comp.first.name.c_str());
                #endif

                (*out_results)[i].components[j].ingredientName = name_copy;
                (*out_results)[i].components[j].proportion = comp.second;
            }
        }
    } catch (const std::exception&) {
        // La variable 'e' no se usa, pero se captura la excepción.
        return -3; // Error interno en el cálculo
    }
    return 0; // Éxito
}

API_FUNCTION void free_diet_results(C_DietResult* results, int results_count) {
    if (!results) return;
    for (int i = 0; i < results_count; ++i) {
        for (int j = 0; j < results[i].componentsCount; ++j) {
            delete[] results[i].components[j].ingredientName;
        }
        delete[] results[i].components;
    }
    delete[] results;
}
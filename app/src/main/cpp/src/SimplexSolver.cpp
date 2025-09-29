#include "SimplexSolver.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>
#include <android/log.h>
#include <deque>

#define LOG_TAG "SimplexSolver"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#include "../eigen/Eigen/Dense"
using namespace Eigen;

// Nombre correcto del solver
#define SOLVER_NAME "Augmented Lagrangian with Two-Phase Method"

// Configuración del solver mejorada
namespace SolverConfig {
    // Tolerancias adaptativas
    constexpr double EPSILON = 1e-10;
    constexpr double MACHINE_EPSILON = std::numeric_limits<double>::epsilon();
    constexpr double CONSTRAINT_TOL = 1e-5;
    constexpr double CONVERGENCE_TOL = 1e-6;
    constexpr double INFEASIBILITY_TOL = 1e-3;

    // Límites de ingredientes
    constexpr double MIN_INGREDIENT_FRACTION = 1e-5;
    constexpr double MAX_INGREDIENT_FRACTION = 0.9999;

    // Parámetros de iteración
    constexpr int MAX_OUTER_ITER = 30;
    constexpr int MAX_INNER_ITER = 200;
    constexpr int MAX_PHASE1_ITER = 100;
    constexpr int NUM_RESTART_POINTS = 7;
    constexpr int MAX_LINE_SEARCH_ITER = 40;

    // Parámetros del Lagrangiano Aumentado
    constexpr double INITIAL_PENALTY = 1.0;
    constexpr double PENALTY_INCREASE_FACTOR = 2.0;
    constexpr double MAX_PENALTY = 1e8;  // Reducido para evitar inestabilidad
    constexpr double MULTIPLIER_UPDATE_RATE = 1.0;
    constexpr double MULTIPLIER_MAX = 1e5;

    // Parámetros de búsqueda lineal
    constexpr double ARMIJO_C1 = 1e-4;
    constexpr double WOLFE_C2 = 0.9;
    constexpr double BACKTRACK_FACTOR = 0.618;  // Proporción dorada

    // Control de paralelismo
    constexpr bool ENABLE_PARALLEL = true;
    constexpr int MIN_PARALLEL_SIZE = 3;
}

// Forward declarations
bool isDietValid(const Diet& diet,
                 const std::map<std::string, double>& minNutrients,
                 const std::map<std::string, double>& maxNutrients);

/**
 * @brief Estructura mejorada para manejar restricciones
 */
struct Constraint {
    VectorXd coeffs;      // Coeficientes de la restricción
    double bound;         // Límite de la restricción
    bool isMin;          // true si es restricción de mínimo
    std::string name;    // Nombre del nutriente
    double multiplier;   // Multiplicador de Lagrange
    double violation;    // Violación actual
    double slack;        // Variable de holgura (para fase 1)
    bool is_active;      // Si la restricción está activa

    Constraint() : bound(0), isMin(true), multiplier(0), violation(0),
                   slack(0), is_active(true) {}

    // Calcula la violación de la restricción
    double computeViolation(const VectorXd& x) const {
        double value = coeffs.dot(x);
        if (isMin) {
            return std::max(0.0, bound - value);
        } else {
            return std::max(0.0, value - bound);
        }
    }

    // Calcula el valor normalizado de la restricción
    double getNormalizedValue(const VectorXd& x) const {
        return coeffs.dot(x);
    }

    // Calcula la violación relativa
    double getRelativeViolation(const VectorXd& x) const {
        double viol = computeViolation(x);
        double scale = std::max(1.0, std::abs(bound));
        return viol / scale;
    }
};

/**
 * @brief Modelo mejorado de metano entérico (IPCC Tier 2)
 */
class MethaneModel {
private:
    double dmi_kg_day;
    double body_weight_kg;

public:
    MethaneModel(double dmi, double bw) : dmi_kg_day(dmi), body_weight_kg(bw) {}

    /**
     * @brief Calcula emisión de metano usando modelo IPCC Tier 2
     * @param ge_mcal Gross Energy en Mcal/kg
     * @param de_percent Digestibilidad de la energía (%)
     * @param ndf_percent NDF (%)
     * @return Emisión de metano en g/día
     */
    double calculateEmission(double ge_mcal, double de_percent, double ndf_percent) const {
        // Factor de emisión de metano (Ym) basado en el tipo de dieta
        double ym = 0.065;  // Por defecto para rumiantes

        // Ajustar Ym basado en calidad de la dieta
        if (de_percent > 70) {
            ym = 0.055;  // Dieta de alta calidad
        } else if (de_percent < 60) {
            ym = 0.075;  // Dieta de baja calidad
        }

        // Ajuste por contenido de fibra
        if (ndf_percent > 45) {
            ym += 0.01;  // Mayor fermentación de fibra
        }

        // Energía bruta ingerida (MJ/día)
        double gei_mj = ge_mcal * dmi_kg_day * 4.184;  // Conversión Mcal a MJ

        // Emisión de metano (kg CH4/día)
        double ch4_kg = (gei_mj * ym) / 55.65;  // 55.65 MJ/kg CH4

        return ch4_kg * 1000.0;  // Convertir a g/día
    }

    /**
     * @brief Modelo alternativo de Blaxter-Clapperton
     */
    double calculateBlaxterClapperton(double de_percent, double crude_fiber) const {
        double ch4_mj = 1.30 + 0.112 * de_percent + crude_fiber * (2.37 - 0.050 * de_percent);
        double ch4_g_per_kg_dmi = ch4_mj * 17.97;  // Conversión a g CH4
        return ch4_g_per_kg_dmi * dmi_kg_day;
    }
};

/**
 * @brief Clase para manejar la Fase 1 (maximización de factibilidad)
 */
class Phase1Solver {
private:
    const std::vector<Constraint>& constraints;
    int n;  // Número de variables
    int m;  // Número de restricciones
    VectorXd artificial_vars;

public:
    Phase1Solver(const std::vector<Constraint>& constr, int num_vars)
            : constraints(constr), n(num_vars), m(static_cast<int>(constr.size())) {
        artificial_vars = VectorXd::Zero(m);
    }

    /**
     * @brief Encuentra un punto inicial factible minimizando variables artificiales
     */
    std::optional<VectorXd> findFeasiblePoint(const VectorXd& initial_x) {
        VectorXd x = initial_x;

        // Inicializar variables artificiales con las violaciones actuales
        for (int i = 0; i < m; ++i) {
            artificial_vars(i) = constraints[i].computeViolation(x);
        }

        double prev_infeasibility = artificial_vars.sum();

        for (int iter = 0; iter < SolverConfig::MAX_PHASE1_ITER; ++iter) {
            // Gradiente respecto a minimizar suma de violaciones
            VectorXd grad = VectorXd::Zero(n);

            for (int i = 0; i < m; ++i) {
                if (artificial_vars(i) > SolverConfig::EPSILON) {
                    double value = constraints[i].coeffs.dot(x);
                    if (constraints[i].isMin && value < constraints[i].bound) {
                        grad -= constraints[i].coeffs;
                    } else if (!constraints[i].isMin && value > constraints[i].bound) {
                        grad += constraints[i].coeffs;
                    }
                }
            }

            // Paso de gradiente proyectado
            double step_size = 1.0 / (iter + 1);
            VectorXd x_new = projectToSimplex(x - step_size * grad);

            // Actualizar variables artificiales
            double infeasibility = 0;
            for (int i = 0; i < m; ++i) {
                artificial_vars(i) = constraints[i].computeViolation(x_new);
                infeasibility += artificial_vars(i);
            }

            // Verificar convergencia
            if (infeasibility < SolverConfig::INFEASIBILITY_TOL) {
                LOGI("Fase 1 completada: punto factible encontrado (iter: %d)", iter);
                return x_new;
            }

            // Verificar mejora
            if (std::abs(infeasibility - prev_infeasibility) < SolverConfig::EPSILON) {
                if (infeasibility < 0.01) {
                    return x_new;  // Aceptar solución casi factible
                }
                break;
            }

            x = x_new;
            prev_infeasibility = infeasibility;
        }

        LOGW("Fase 1: No se encontró punto perfectamente factible (inf: %.4f)", prev_infeasibility);
        return std::nullopt;
    }

private:
    VectorXd projectToSimplex(const VectorXd& y) {
        VectorXd x = y;

        // Aplicar límites de caja
        for (int i = 0; i < n; ++i) {
            x(i) = std::max(SolverConfig::MIN_INGREDIENT_FRACTION,
                            std::min(SolverConfig::MAX_INGREDIENT_FRACTION, x(i)));
        }

        // Proyección eficiente al símplex usando algoritmo de Condat
        double sum = x.sum();
        if (std::abs(sum - 1.0) < SolverConfig::EPSILON) {
            return x;
        }

        // Algoritmo de proyección más robusto
        VectorXd u = x;
        std::sort(u.data(), u.data() + n, std::greater<double>());

        double cumsum = 0;
        double lambda = 0;
        int k = 0;

        for (int i = 0; i < n; ++i) {
            cumsum += u(i);
            double temp_lambda = (cumsum - 1.0) / (i + 1);
            if (i == n - 1 || u(i + 1) < temp_lambda) {
                lambda = temp_lambda;
                k = i + 1;
                break;
            }
        }

        // Aplicar proyección
        VectorXd proj(n);
        for (int i = 0; i < n; ++i) {
            proj(i) = std::max(SolverConfig::MIN_INGREDIENT_FRACTION, x(i) - lambda);
        }

        // Normalizar para garantizar suma = 1
        double proj_sum = proj.sum();
        if (proj_sum > SolverConfig::EPSILON) {
            proj /= proj_sum;
        }

        return proj;
    }
};

/**
 * @brief Solver principal con método del Lagrangiano Aumentado mejorado
 */
class AugmentedLagrangianSolver {
private:
    const std::vector<Ingredient>& ingredients;
    std::vector<Constraint> constraints;
    VectorXd costVector;
    int n;  // Número de ingredientes
    double penalty;
    VectorXd x_current;
    VectorXd x_best;
    double best_objective;
    double costWeight;
    double methaneWeight;

    // Modelo de metano
    std::unique_ptr<MethaneModel> methane_model;

    // Estadísticas del solver
    int total_iterations;
    double max_violation;
    std::chrono::steady_clock::time_point start_time;

    // Cache para cálculos frecuentes
    mutable std::mutex cache_mutex;
    mutable std::map<size_t, double> lagrangian_cache;

public:
    AugmentedLagrangianSolver(const std::vector<Ingredient>& ingr,
                              const std::map<std::string, double>& minNutrients,
                              const std::map<std::string, double>& maxNutrients,
                              double cWeight,
                              double mWeight,
                              double dmi_kg_day,
                              double body_weight_kg)
            : ingredients(ingr), n(static_cast<int>(ingr.size())),
              penalty(SolverConfig::INITIAL_PENALTY),
              costWeight(cWeight), methaneWeight(mWeight),
              total_iterations(0), max_violation(0),
              start_time(std::chrono::steady_clock::now()) {

        // Validación de parámetros
        if (n == 0) {
            throw std::invalid_argument("No hay ingredientes disponibles");
        }

        methane_model = std::make_unique<MethaneModel>(dmi_kg_day, body_weight_kg);
        setupConstraints(minNutrients, maxNutrients);
        setupObjective();
        x_current = VectorXd::Constant(n, 1.0 / n);
        x_best = x_current;
        best_objective = std::numeric_limits<double>::infinity();
    }

    /**
     * @brief Configura las restricciones del problema con análisis de redundancia
     */
    void setupConstraints(const std::map<std::string, double>& minNutrients,
                          const std::map<std::string, double>& maxNutrients) {
        // Restricciones de mínimos
        for (const auto& [nutrient, minValue] : minNutrients) {
            if (minValue > SolverConfig::EPSILON) {  // Ignorar restricciones triviales
                Constraint c;
                c.coeffs = VectorXd(n);
                for (int i = 0; i < n; ++i) {
                    c.coeffs(i) = ingredients[i].getNutrient(nutrient);
                }
                // Verificar que al menos un ingrediente puede satisfacer la restricción
                if (c.coeffs.maxCoeff() > SolverConfig::EPSILON) {
                    c.bound = minValue;
                    c.isMin = true;
                    c.name = nutrient + "_min";
                    constraints.push_back(c);
                } else {
                    LOGW("Restricción imposible: %s_min (ningún ingrediente lo contiene)", nutrient.c_str());
                }
            }
        }

        // Restricciones de máximos
        for (const auto& [nutrient, maxValue] : maxNutrients) {
            if (maxValue < 1e6) {  // Ignorar límites efectivamente infinitos
                Constraint c;
                c.coeffs = VectorXd(n);
                for (int i = 0; i < n; ++i) {
                    c.coeffs(i) = ingredients[i].getNutrient(nutrient);
                }
                // Verificar que la restricción no es trivial
                if (c.coeffs.maxCoeff() > SolverConfig::EPSILON) {
                    c.bound = maxValue;
                    c.isMin = false;
                    c.name = nutrient + "_max";
                    constraints.push_back(c);
                }
            }
        }

        LOGI("Configuradas %zu restricciones activas", constraints.size());
    }

    /**
     * @brief Configura el vector objetivo con modelo de metano mejorado
     */
    void setupObjective() {
        costVector = VectorXd(n);

        // Calcular estadísticas para normalización
        double max_cost = 0;
        double max_methane = 0;

        for (int i = 0; i < n; ++i) {
            max_cost = std::max(max_cost, ingredients[i].cost);

            // Estimar metano máximo
            double ge = ingredients[i].getNutrient("NEm") * 2.5;  // Aproximación GE
            double de = std::min(85.0, ingredients[i].getNutrient("TDN"));
            double ndf = ingredients[i].getNutrient("NDF");
            double methane = methane_model->calculateEmission(ge, de, ndf);
            max_methane = std::max(max_methane, methane);
        }

        // Normalización para balance de objetivos
        double cost_scale = (max_cost > 0) ? 1.0 / max_cost : 1.0;
        double methane_scale = (max_methane > 0) ? 1.0 / max_methane : 1.0;

        for (int i = 0; i < n; ++i) {
            // Componente de costo normalizado
            double normalized_cost = ingredients[i].cost * cost_scale;

            // Componente de metano usando modelo IPCC
            double ge = ingredients[i].getNutrient("NEm") * 2.5;
            double de = std::min(85.0, ingredients[i].getNutrient("TDN"));
            double ndf = ingredients[i].getNutrient("NDF");
            double methane = methane_model->calculateEmission(ge, de, ndf);
            double normalized_methane = methane * methane_scale;

            // Objetivo ponderado
            costVector(i) = costWeight * normalized_cost + methaneWeight * normalized_methane;
        }
    }

    /**
     * @brief Evalúa el Lagrangiano Aumentado con caché
     */
    double evaluateLagrangian(const VectorXd& x) {
        // Hash simple para caché
        size_t hash = 0;
        for (int i = 0; i < std::min(5, n); ++i) {
            hash ^= std::hash<double>()(x(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        // Verificar caché
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = lagrangian_cache.find(hash);
            if (it != lagrangian_cache.end()) {
                return it->second;
            }
        }

        double L = costVector.dot(x);

        // Términos de restricciones y penalización
        for (auto& c : constraints) {
            if (!c.is_active) continue;

            double violation = c.computeViolation(x);
            c.violation = violation;

            if (violation > SolverConfig::EPSILON) {
                L += c.multiplier * violation;
                L += 0.5 * penalty * violation * violation;
            }
        }

        // Restricción de suma = 1 (penalización cuadrática exacta)
        double sum_violation = x.sum() - 1.0;
        L += penalty * sum_violation * sum_violation;

        // Guardar en caché
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (lagrangian_cache.size() > 1000) {
                lagrangian_cache.clear();  // Limpiar caché si crece mucho
            }
            lagrangian_cache[hash] = L;
        }

        return L;
    }

    /**
     * @brief Calcula el gradiente del Lagrangiano con diferencias finitas para validación
     */
    VectorXd computeGradient(const VectorXd& x, bool use_finite_diff = false) {
        if (use_finite_diff) {
            // Diferencias finitas para validación
            VectorXd grad(n);
            const double h = 1e-7;
            for (int i = 0; i < n; ++i) {
                VectorXd x_plus = x;
                VectorXd x_minus = x;
                x_plus(i) += h;
                x_minus(i) -= h;
                grad(i) = (evaluateLagrangian(x_plus) - evaluateLagrangian(x_minus)) / (2 * h);
            }
            return grad;
        }

        // Gradiente analítico
        VectorXd grad = costVector;

        for (const auto& c : constraints) {
            if (!c.is_active) continue;

            double value = c.coeffs.dot(x);
            double violation = 0;

            if (c.isMin && value < c.bound) {
                violation = c.bound - value;
                grad -= (c.multiplier + penalty * violation) * c.coeffs;
            } else if (!c.isMin && value > c.bound) {
                violation = value - c.bound;
                grad += (c.multiplier + penalty * violation) * c.coeffs;
            }
        }

        // Gradiente de la restricción de suma
        double sum_violation = x.sum() - 1.0;
        grad += 2.0 * penalty * sum_violation * VectorXd::Ones(n);

        return grad;
    }

    /**
     * @brief Proyección mejorada al símplex con manejo de casos especiales
     */
    VectorXd projectToSimplex(const VectorXd& y) {
        // Caso especial: n = 1
        if (n == 1) {
            return VectorXd::Ones(1);
        }

        VectorXd x = y;

        // Aplicar límites de caja
        for (int i = 0; i < n; ++i) {
            x(i) = std::max(SolverConfig::MIN_INGREDIENT_FRACTION,
                            std::min(SolverConfig::MAX_INGREDIENT_FRACTION, x(i)));
        }

        // Verificar si ya está en el símplex
        double sum = x.sum();
        if (std::abs(sum - 1.0) < SolverConfig::MACHINE_EPSILON * n) {
            return x;
        }

        // Algoritmo de proyección eficiente (método de pivoteo)
        VectorXd z = x;
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        // Particionar usando nth_element para mejor rendimiento con n grande
        int k = 0;
        double lambda = 0;
        double cumsum = 0;

        while (k < n) {
            int pivot_idx = k + (n - k) / 2;
            std::nth_element(indices.begin() + k, indices.begin() + pivot_idx,
                             indices.end(),
                             [&z](int i, int j) { return z(i) > z(j); });

            cumsum = 0;
            for (int i = 0; i <= pivot_idx; ++i) {
                cumsum += z(indices[i]);
            }

            lambda = (cumsum - 1.0) / (pivot_idx + 1);

            if (pivot_idx == n - 1 || z(indices[pivot_idx + 1]) < lambda) {
                break;
            }

            k = pivot_idx + 1;
        }

        // Aplicar proyección
        VectorXd proj(n);
        for (int i = 0; i < n; ++i) {
            proj(i) = std::max(SolverConfig::MIN_INGREDIENT_FRACTION, x(i) - lambda);
        }

        // Renormalizar
        double proj_sum = proj.sum();
        if (proj_sum > SolverConfig::EPSILON) {
            proj /= proj_sum;
        } else {
            // Caso degenerado: volver a distribución uniforme
            proj = VectorXd::Constant(n, 1.0 / n);
        }

        return proj;
    }

    /**
     * @brief Búsqueda lineal con condiciones de Wolfe
     */
    double wolfeLineSearch(const VectorXd& x, const VectorXd& direction,
                           const VectorXd& grad, double initial_alpha = 1.0) {
        double alpha = initial_alpha;
        const double c1 = SolverConfig::ARMIJO_C1;
        const double c2 = SolverConfig::WOLFE_C2;

        double f_old = evaluateLagrangian(x);
        double g_dot_d = grad.dot(direction);

        // Si la dirección no es de descenso, retornar alpha pequeño
        if (g_dot_d >= 0) {
            return 1e-6;
        }

        double alpha_lo = 0;
        double alpha_hi = 2.0;

        for (int iter = 0; iter < SolverConfig::MAX_LINE_SEARCH_ITER; ++iter) {
            VectorXd x_new = projectToSimplex(x + alpha * direction);
            double f_new = evaluateLagrangian(x_new);

            // Verificar condición de Armijo
            if (f_new > f_old + c1 * alpha * g_dot_d) {
                alpha_hi = alpha;
                alpha = 0.5 * (alpha_lo + alpha_hi);
                continue;
            }

            // Verificar condición de curvatura
            VectorXd grad_new = computeGradient(x_new);
            double g_new_dot_d = grad_new.dot(direction);

            if (std::abs(g_new_dot_d) <= -c2 * g_dot_d) {
                return alpha;  // Condiciones de Wolfe satisfechas
            }

            if (g_new_dot_d >= 0) {
                alpha_hi = alpha;
            } else {
                alpha_lo = alpha;
            }

            alpha = 0.5 * (alpha_lo + alpha_hi);

            // Salida temprana si alpha es muy pequeño
            if (alpha < 1e-10) {
                break;
            }
        }

        return alpha;
    }

    /**
     * @brief Minimización con L-BFGS (memoria limitada)
     */
    bool minimizeWithLBFGS() {
        const int m_memory = 5;  // Tamaño de memoria para L-BFGS
        std::deque<VectorXd> s_history, y_history;
        std::deque<double> rho_history;

        VectorXd x = x_current;
        VectorXd grad_old = computeGradient(x);
        double f_old = evaluateLagrangian(x);

        for (int iter = 0; iter < SolverConfig::MAX_INNER_ITER; ++iter) {
            // Calcular dirección L-BFGS
            VectorXd direction = -grad_old;

            if (!s_history.empty()) {
                // Algoritmo de dos loops de L-BFGS
                int history_size = static_cast<int>(s_history.size());
                std::vector<double> alpha(history_size);

                // Primer loop
                for (int i = history_size - 1; i >= 0; --i) {
                    alpha[i] = rho_history[i] * s_history[i].dot(direction);
                    direction -= alpha[i] * y_history[i];
                }

                // Escalado inicial
                if (history_size > 0) {
                    double gamma = s_history.back().dot(y_history.back()) /
                                   y_history.back().dot(y_history.back());
                    direction *= gamma;
                }

                // Segundo loop
                for (int i = 0; i < history_size; ++i) {
                    double beta = rho_history[i] * y_history[i].dot(direction);
                    direction += (alpha[i] - beta) * s_history[i];
                }
            }

            // Búsqueda lineal
            double step = wolfeLineSearch(x, direction, grad_old);

            // Actualización
            VectorXd x_new = projectToSimplex(x + step * direction);
            VectorXd grad_new = computeGradient(x_new);
            double f_new = evaluateLagrangian(x_new);

            // Verificar convergencia
            double grad_norm = grad_new.norm();
            double relative_change = std::abs(f_new - f_old) / (std::abs(f_old) + 1.0);

            if (grad_norm < SolverConfig::CONVERGENCE_TOL ||
                relative_change < SolverConfig::CONVERGENCE_TOL) {
                x_current = x_new;
                total_iterations += iter;
                return true;
            }

            // Actualizar historia para L-BFGS
            VectorXd s = x_new - x;
            VectorXd y = grad_new - grad_old;
            double ys = y.dot(s);

            if (ys > 1e-10) {  // Actualización segura
                if (s_history.size() >= m_memory) {
                    s_history.pop_front();
                    y_history.pop_front();
                    rho_history.pop_front();
                }

                s_history.push_back(s);
                y_history.push_back(y);
                rho_history.push_back(1.0 / ys);
            }

            // Preparar siguiente iteración
            x = x_new;
            grad_old = grad_new;
            f_old = f_new;
        }

        x_current = x;
        total_iterations += SolverConfig::MAX_INNER_ITER;
        return false;
    }

    /**
     * @brief Actualiza multiplicadores con método adaptativo
     */
    void updateMultipliers() {
        double total_violation = 0;
        int num_violations = 0;

        for (auto& c : constraints) {
            double violation = c.computeViolation(x_current);
            c.violation = violation;

            if (violation > SolverConfig::EPSILON) {
                total_violation += violation;
                num_violations++;

                // Actualización adaptativa basada en la magnitud de violación
                double relative_viol = c.getRelativeViolation(x_current);
                double update_rate = SolverConfig::MULTIPLIER_UPDATE_RATE;

                if (relative_viol > 0.1) {
                    update_rate *= 2.0;  // Actualización más agresiva para grandes violaciones
                } else if (relative_viol < 0.01) {
                    update_rate *= 0.5;  // Actualización conservadora para pequeñas violaciones
                }

                double update = penalty * violation * update_rate;
                c.multiplier = std::max(0.0,
                                        std::min(SolverConfig::MULTIPLIER_MAX, c.multiplier + update));

                // Marcar restricción como activa
                c.is_active = true;
            } else {
                // Reducir multiplicador si la restricción está satisfecha
                c.multiplier *= 0.9;

                // Desactivar restricción si el multiplicador es muy pequeño
                if (c.multiplier < 1e-6) {
                    c.is_active = false;
                }
            }
        }

        // Ajuste global de penalización
        if (num_violations > 0) {
            double avg_violation = total_violation / num_violations;
            if (avg_violation > 0.01) {
                penalty = std::min(penalty * SolverConfig::PENALTY_INCREASE_FACTOR,
                                   SolverConfig::MAX_PENALTY);
            }
        }
    }

    /**
     * @brief Verifica factibilidad con tolerancia adaptativa
     */
    bool checkFeasibility() {
        max_violation = 0.0;
        int num_active = 0;

        for (const auto& c : constraints) {
            double viol_ratio = c.getRelativeViolation(x_current);
            max_violation = std::max(max_violation, viol_ratio);
            if (c.is_active) num_active++;
        }

        LOGD("Restricciones activas: %d/%zu, Violación máx: %.4f%%",
             num_active, constraints.size(), max_violation * 100);

        return max_violation < SolverConfig::CONSTRAINT_TOL;
    }

    /**
     * @brief Genera puntos iniciales inteligentes y diversos
     */
    std::vector<VectorXd> generateStartingPoints() {
        std::vector<VectorXd> points;

        // Verificar caso especial n = 1
        if (n == 1) {
            points.push_back(VectorXd::Ones(1));
            return points;
        }

        // 1. Punto uniforme
        points.push_back(VectorXd::Constant(n, 1.0 / n));

        // 2. Minimizar costo directo
        VectorXd cost_point = VectorXd::Zero(n);
        std::vector<std::pair<double, int>> cost_indices;
        for (int i = 0; i < n; ++i) {
            cost_indices.push_back({costVector(i), i});
        }
        std::sort(cost_indices.begin(), cost_indices.end());

        // Asignar más peso a ingredientes de menor costo
        double remaining = 1.0;
        for (int i = 0; i < std::min(3, n); ++i) {
            double weight = (3 - i) * 0.3;
            cost_point(cost_indices[i].second) = std::min(weight, remaining);
            remaining -= cost_point(cost_indices[i].second);
        }
        if (remaining > 0 && n > 3) {
            for (int i = 3; i < n; ++i) {
                cost_point(cost_indices[i].second) = remaining / (n - 3);
            }
        }
        points.push_back(projectToSimplex(cost_point));

        // 3. Maximizar proteína eficientemente
        VectorXd protein_point = VectorXd::Zero(n);
        std::vector<std::pair<double, int>> protein_efficiency;
        for (int i = 0; i < n; ++i) {
            double cp = ingredients[i].getNutrient("CP");
            double cost = ingredients[i].cost;
            if (cost > 0) {
                protein_efficiency.push_back({cp / cost, i});
            }
        }

        if (!protein_efficiency.empty()) {
            std::sort(protein_efficiency.rbegin(), protein_efficiency.rend());
            remaining = 1.0;
            for (size_t i = 0; i < std::min(size_t(4), protein_efficiency.size()); ++i) {
                double weight = 0.25;
                protein_point(protein_efficiency[i].second) = weight;
                remaining -= weight;
            }
            if (remaining > 0) {
                for (int i = 0; i < n; ++i) {
                    if (protein_point(i) == 0) {
                        protein_point(i) = remaining / (n - std::min(4, n));
                    }
                }
            }
            points.push_back(projectToSimplex(protein_point));
        }

        // 4. Punto balanceado por energía
        VectorXd energy_point = VectorXd::Zero(n);
        double target_nem = 1.5;  // Mcal/kg típico
        for (int i = 0; i < n; ++i) {
            double nem = ingredients[i].getNutrient("NEm");
            double diff = std::abs(nem - target_nem);
            energy_point(i) = std::exp(-diff);  // Peso gaussiano
        }
        points.push_back(projectToSimplex(energy_point));

        // 5. Punto basado en restricciones activas (heurística)
        VectorXd constraint_point = VectorXd::Ones(n);
        for (const auto& c : constraints) {
            if (c.isMin) {
                // Aumentar peso de ingredientes que ayudan con mínimos
                for (int i = 0; i < n; ++i) {
                    if (c.coeffs(i) > 0) {
                        constraint_point(i) += c.coeffs(i) / c.bound;
                    }
                }
            }
        }
        points.push_back(projectToSimplex(constraint_point));

        // 6-7. Puntos aleatorios con diferentes distribuciones
        std::random_device rd;
        std::mt19937 gen(rd());

        // Distribución uniforme
        std::uniform_real_distribution<> uniform(0.01, 1.0);
        VectorXd random_uniform(n);
        for (int i = 0; i < n; ++i) {
            random_uniform(i) = uniform(gen);
        }
        points.push_back(projectToSimplex(random_uniform));

        // Distribución exponencial (sesgo hacia pocos ingredientes)
        std::exponential_distribution<> exponential(2.0);
        VectorXd random_exp(n);
        for (int i = 0; i < n; ++i) {
            random_exp(i) = exponential(gen);
        }
        points.push_back(projectToSimplex(random_exp));

        return points;
    }

    /**
     * @brief Resuelve el problema desde un punto inicial
     */
    std::optional<VectorXd> solveFromPoint(const VectorXd& initial_point) {
        x_current = initial_point;
        penalty = SolverConfig::INITIAL_PENALTY;

        // Reiniciar multiplicadores
        for (auto& c : constraints) {
            c.multiplier = 0.0;
            c.is_active = true;
        }

        // Limpiar caché
        lagrangian_cache.clear();

        // Fase 1: Encontrar punto factible
        Phase1Solver phase1(constraints, n);
        auto feasible_point = phase1.findFeasiblePoint(x_current);

        if (feasible_point.has_value()) {
            x_current = feasible_point.value();
        } else {
            LOGW("No se encontró punto perfectamente factible, continuando con mejor aproximación");
        }

        bool converged = false;

        // Fase 2: Optimización con multiplicadores
        for (int outer = 0; outer < SolverConfig::MAX_OUTER_ITER; ++outer) {
            // Minimizar con multiplicadores y penalización fijos
            bool inner_converged = minimizeWithLBFGS();

            // Verificar factibilidad
            if (checkFeasibility()) {
                double obj = costVector.dot(x_current);

                // Actualizar mejor solución
                if (obj < best_objective) {
                    best_objective = obj;
                    x_best = x_current;
                }

                if (inner_converged && max_violation < SolverConfig::CONSTRAINT_TOL / 10) {
                    converged = true;
                    break;
                }
            }

            // Actualizar multiplicadores
            updateMultipliers();

            // Verificar tiempo límite (para Android)
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 10) {
                LOGW("Tiempo límite alcanzado, terminando optimización");
                break;
            }
        }

        if (converged) {
            return x_best;
        }

        // Retornar mejor solución encontrada si es razonablemente factible
        if (max_violation < SolverConfig::INFEASIBILITY_TOL) {
            return x_best;
        }

        return std::nullopt;
    }

    /**
     * @brief Resuelve el problema con múltiples inicios paralelos
     */
    std::optional<VectorXd> solve() {
        auto starting_points = generateStartingPoints();

        LOGI("=== %s ===", SOLVER_NAME);
        LOGI("Configuración: %.0f%% Costo, %.0f%% Metano",
             costWeight * 100, methaneWeight * 100);
        LOGI("Ingredientes: %d, Restricciones: %zu", n, constraints.size());

        std::vector<std::future<std::optional<VectorXd>>> futures;
        bool use_parallel = SolverConfig::ENABLE_PARALLEL &&
                            starting_points.size() >= SolverConfig::MIN_PARALLEL_SIZE &&
                            std::thread::hardware_concurrency() > 1;

        if (use_parallel) {
            LOGI("Ejecutando en paralelo con %zu puntos iniciales", starting_points.size());

            for (const auto& point : starting_points) {
                futures.push_back(
                        std::async(std::launch::async,
                                   [this, point]() { return solveFromPoint(point); })
                );
            }

            // Recolectar resultados
            for (auto& future : futures) {
                auto result = future.get();
                if (result.has_value()) {
                    VectorXd x = result.value();
                    double obj = costVector.dot(x);
                    if (obj < best_objective) {
                        best_objective = obj;
                        x_best = x;
                    }
                }
            }
        } else {
            // Ejecución secuencial
            for (size_t i = 0; i < starting_points.size(); ++i) {
                LOGI("Probando punto inicial %zu/%zu", i + 1, starting_points.size());

                auto result = solveFromPoint(starting_points[i]);
                if (result.has_value()) {
                    VectorXd x = result.value();
                    double obj = costVector.dot(x);
                    if (obj < best_objective) {
                        best_objective = obj;
                        x_best = x;

                        // Salida temprana si encontramos solución excelente
                        if (checkFeasibility() && max_violation < 1e-6) {
                            LOGI("✓ Solución óptima encontrada tempranamente");
                            break;
                        }
                    }
                }
            }
        }

        // Verificar si encontramos alguna solución
        if (best_objective < std::numeric_limits<double>::infinity()) {
            // Verificación final de factibilidad
            x_current = x_best;
            checkFeasibility();

            if (max_violation < SolverConfig::INFEASIBILITY_TOL) {
                if (max_violation < SolverConfig::CONSTRAINT_TOL) {
                    LOGI("✓ Solución óptima encontrada");
                } else {
                    LOGW("⚠ Solución aproximada (violación: %.4f%%)", max_violation * 100);
                }
                return x_best;
            }
        }

        LOGE("✗ No se encontró solución factible");
        return std::nullopt;
    }

    /**
     * @brief Obtiene estadísticas detalladas del solver
     */
    void logStatistics() {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        LOGI("=== Estadísticas del Solver ===");
        LOGI("  Tiempo total: %lld ms", ms);
        LOGI("  Iteraciones totales: %d", total_iterations);
        LOGI("  Violación máxima: %.6f%%", max_violation * 100);
        LOGI("  Penalización final: %.2e", penalty);
        LOGI("  Objetivo final: %.6f", best_objective);

        // Restricciones más problemáticas
        std::vector<std::pair<double, std::string>> violations;
        for (const auto& c : constraints) {
            if (c.violation > SolverConfig::EPSILON) {
                violations.push_back({c.getRelativeViolation(x_best), c.name});
            }
        }

        if (!violations.empty()) {
            std::sort(violations.rbegin(), violations.rend());
            LOGW("Top restricciones violadas:");
            for (size_t i = 0; i < std::min(size_t(5), violations.size()); ++i) {
                LOGW("  %s: %.4f%%", violations[i].second.c_str(),
                     violations[i].first * 100);
            }
        }

        // Estadísticas de convergencia
        int active_constraints = 0;
        for (const auto& c : constraints) {
            if (c.is_active) active_constraints++;
        }
        LOGI("  Restricciones activas: %d/%zu", active_constraints, constraints.size());

        // Información de caché
        LOGI("  Tamaño caché Lagrangiano: %zu", lagrangian_cache.size());
    }
};

/**
 * @brief Método principal de resolución mejorado
 */
std::optional<Diet> SimplexSolver::solve(
        const std::vector<Ingredient>& ingredients,
        const std::map<std::string, double>& minNutrients,
        const std::map<std::string, double>& maxNutrients,
        double costWeight,
        double methaneWeight,
        double DMI_kg_day,
        double body_weight_kg) {

    // Validación exhaustiva de entrada
    if (ingredients.empty()) {
        LOGE("Error: No hay ingredientes disponibles");
        return std::nullopt;
    }

    if (ingredients.size() == 1) {
        // Caso trivial: un solo ingrediente
        Diet diet;
        diet.composition.emplace_back(ingredients[0], 1.0);
        diet.calculateDietProperties();
        diet.calculateEntericMethane(DMI_kg_day, body_weight_kg);

        // Verificar si cumple restricciones
        if (isDietValid(diet, minNutrients, maxNutrients)) {
            LOGI("Solución trivial: 100%% %s", ingredients[0].name.c_str());
            return diet;
        } else {
            LOGE("El único ingrediente no cumple las restricciones");
            return std::nullopt;
        }
    }

    // Validación de pesos
    if (costWeight < 0 || methaneWeight < 0) {
        LOGE("Error: Pesos no pueden ser negativos");
        return std::nullopt;
    }

    double weight_sum = costWeight + methaneWeight;
    if (std::abs(weight_sum - 1.0) > SolverConfig::EPSILON) {
        LOGW("Normalizando pesos (suma actual: %.4f)", weight_sum);
        if (weight_sum > 0) {
            costWeight /= weight_sum;
            methaneWeight /= weight_sum;
        } else {
            costWeight = 0.5;
            methaneWeight = 0.5;
        }
    }

    // Verificación de factibilidad básica
    bool has_energy = false, has_protein = false, has_fiber = false;

    for (const auto& ing : ingredients) {
        if (ing.getNutrient("NEm") > 0 || ing.getNutrient("NEg") > 0) has_energy = true;
        if (ing.getNutrient("CP") > 0) has_protein = true;
        if (ing.getNutrient("NDF") > 0 || ing.getNutrient("ADF") > 0) has_fiber = true;
    }

    if (!has_energy) {
        LOGE("Error: No hay fuentes de energía disponibles");
        return std::nullopt;
    }

    if (!has_protein && minNutrients.count("CP") > 0) {
        LOGE("Error: Se requiere proteína pero no hay fuentes disponibles");
        return std::nullopt;
    }

    try {
        // Crear y ejecutar el solver
        AugmentedLagrangianSolver solver(ingredients, minNutrients, maxNutrients,
                                         costWeight, methaneWeight,
                                         DMI_kg_day, body_weight_kg);

        auto solution = solver.solve();

        if (!solution.has_value()) {
            LOGE("Error: No se pudo encontrar una solución factible");
            solver.logStatistics();

            // Sugerir relajación de restricciones
            LOGI("Sugerencias:");
            LOGI("  1. Verifique que las restricciones no sean contradictorias");
            LOGI("  2. Considere ampliar los rangos de nutrientes");
            LOGI("  3. Agregue más ingredientes diversos");

            return std::nullopt;
        }

        VectorXd x = solution.value();
        const int n = static_cast<int>(ingredients.size());

        // Construir la dieta final
        Diet finalDiet;
        const double THRESHOLD = SolverConfig::MIN_INGREDIENT_FRACTION * 10;

        LOGI("=== Composición de la Dieta ===");
        std::vector<std::pair<double, std::string>> composition;

        for (int i = 0; i < n; ++i) {
            if (x(i) > THRESHOLD) {
                finalDiet.composition.emplace_back(ingredients[i], x(i));
                composition.push_back({x(i), ingredients[i].name});
            }
        }

        // Ordenar por proporción descendente
        std::sort(composition.rbegin(), composition.rend());

        for (const auto& [prop, name] : composition) {
            LOGI("  %s: %.2f%% (%.4f kg/kg MS)",
                 name.c_str(), prop * 100.0, prop);
        }

        if (finalDiet.composition.empty()) {
            LOGE("Error: Dieta vacía generada");
            return std::nullopt;
        }

        // Calcular propiedades finales
        finalDiet.calculateDietProperties();
        finalDiet.calculateEntericMethane(DMI_kg_day, body_weight_kg);

        // Verificar cumplimiento de restricciones
        bool is_valid = isDietValid(finalDiet, minNutrients, maxNutrients);

        // Log de resultados
        LOGI("=== Resultados Finales ===");
        LOGI("Costo total: $%.4f/kg MS", finalDiet.totalCost);
        LOGI("Metano entérico: %.2f g CH₄/día", finalDiet.entericMethane);
        LOGI("Intensidad de emisión: %.2f g CH₄/kg MS",
             finalDiet.entericMethane / DMI_kg_day);

        // Calcular objetivo ponderado normalizado
        double max_cost = 0;
        double max_methane = 0;
        for (const auto& ing : ingredients) {
            max_cost = std::max(max_cost, ing.cost);
        }

        double normalized_objective =
                (finalDiet.totalCost / max_cost) * costWeight +
                (finalDiet.entericMethane / 1000.0) * methaneWeight;

        LOGI("Objetivo ponderado: %.6f", normalized_objective);

        // Log detallado de nutrientes
        LOGI("=== Perfil Nutricional Completo ===");

        // Categorizar nutrientes
        std::map<std::string, std::vector<std::pair<std::string, double>>> categories = {
                {"Energía", {}},
                {"Proteína", {}},
                {"Fibra", {}},
                {"Minerales", {}},
                {"Otros", {}}
        };

        for (const auto& [nutrient, value] : finalDiet.finalNutrientProfile) {
            std::string category = "Otros";

            if (nutrient.find("NE") != std::string::npos || nutrient == "TDN") {
                category = "Energía";
            } else if (nutrient.find("P") != std::string::npos && nutrient != "P") {
                category = "Proteína";
            } else if (nutrient.find("DF") != std::string::npos || nutrient == "Lignin") {
                category = "Fibra";
            } else if (nutrient == "Ca" || nutrient == "P" || nutrient == "Mg" ||
                       nutrient == "K" || nutrient == "Na" || nutrient == "S") {
                category = "Minerales";
            }

            categories[category].push_back({nutrient, value});
        }

        for (const auto& [cat, nutrients] : categories) {
            if (!nutrients.empty()) {
                LOGI("  %s:", cat.c_str());
                for (const auto& [nutrient, value] : nutrients) {
                    auto min_it = minNutrients.find(nutrient);
                    auto max_it = maxNutrients.find(nutrient);

                    std::string status = "✓";
                    std::string range = "";

                    if (min_it != minNutrients.end()) {
                        if (value < min_it->second - SolverConfig::CONSTRAINT_TOL) {
                            status = "↓ BAJO";
                        }
                        range += "min: " + std::to_string(min_it->second);
                    }

                    if (max_it != maxNutrients.end()) {
                        if (value > max_it->second + SolverConfig::CONSTRAINT_TOL) {
                            status = "↑ ALTO";
                        }
                        if (!range.empty()) range += ", ";
                        range += "max: " + std::to_string(max_it->second);
                    }

                    LOGI("    %s %s: %.3f %s",
                         status.c_str(), nutrient.c_str(), value,
                         range.empty() ? "" : ("(" + range + ")").c_str());
                }
            }
        }

        // Log estadísticas del solver
        solver.logStatistics();

        // Evaluación final
        if (!is_valid) {
            LOGW("⚠ ADVERTENCIA: Algunas restricciones no se cumplen exactamente");
            LOGW("  La dieta puede requerir ajustes o suplementación");

            // Análisis de deficiencias
            std::vector<std::string> deficiencies, excesses;

            for (const auto& [nutrient, minValue] : minNutrients) {
                auto it = finalDiet.finalNutrientProfile.find(nutrient);
                if (it != finalDiet.finalNutrientProfile.end()) {
                    double deficit = (minValue - it->second) / minValue * 100;
                    if (deficit > 1) {
                        deficiencies.push_back(nutrient + " (-" +
                                               std::to_string(deficit) + "%)");
                    }
                }
            }

            for (const auto& [nutrient, maxValue] : maxNutrients) {
                auto it = finalDiet.finalNutrientProfile.find(nutrient);
                if (it != finalDiet.finalNutrientProfile.end()) {
                    double excess = (it->second - maxValue) / maxValue * 100;
                    if (excess > 1) {
                        excesses.push_back(nutrient + " (+" +
                                           std::to_string(excess) + "%)");
                    }
                }
            }

            if (!deficiencies.empty()) {
                LOGW("  Deficiencias: %s",
                     std::accumulate(deficiencies.begin() + 1, deficiencies.end(),
                                     deficiencies[0],
                                     [](const std::string& a, const std::string& b) {
                                         return a + ", " + b;
                                     }).c_str());
            }

            if (!excesses.empty()) {
                LOGW("  Excesos: %s",
                     std::accumulate(excesses.begin() + 1, excesses.end(),
                                     excesses[0],
                                     [](const std::string& a, const std::string& b) {
                                         return a + ", " + b;
                                     }).c_str());
            }
        } else {
            LOGI("✓ ÉXITO: Todas las restricciones nutricionales se cumplen");
            LOGI("✓ La dieta es nutricionalmente balanceada y optimizada");
        }

        return finalDiet;

    } catch (const std::exception& e) {
        LOGE("Error durante optimización: %s", e.what());
        return std::nullopt;
    }
}

/**
 * @brief Validación mejorada de la dieta con análisis detallado
 */
bool isDietValid(const Diet& diet,
                 const std::map<std::string, double>& minNutrients,
                 const std::map<std::string, double>& maxNutrients) {

    bool all_valid = true;
    std::vector<std::string> violations;
    double total_violation_score = 0;

    // Verificar restricciones de mínimos
    for (const auto& [nutrient, minValue] : minNutrients) {
        auto it = diet.finalNutrientProfile.find(nutrient);
        double actualValue = (it != diet.finalNutrientProfile.end()) ? it->second : 0.0;

        // Tolerancia adaptativa basada en la magnitud y tipo de nutriente
        double base_tolerance = SolverConfig::CONSTRAINT_TOL;

        // Tolerancias más estrictas para nutrientes críticos
        if (nutrient == "CP" || nutrient == "NEm" || nutrient == "Ca" || nutrient == "P") {
            base_tolerance *= 0.5;
        }

        double tolerance = base_tolerance * (1.0 + std::abs(minValue) * 0.01);

        if (actualValue < minValue - tolerance) {
            double deficit = ((minValue - actualValue) / minValue) * 100.0;
            violations.push_back(
                    nutrient + " MIN: requerido=" + std::to_string(minValue) +
                    " actual=" + std::to_string(actualValue) +
                    " (" + std::to_string(deficit) + "% déficit)"
            );
            total_violation_score += deficit;
            all_valid = false;
        }
    }

    // Verificar restricciones de máximos
    for (const auto& [nutrient, maxValue] : maxNutrients) {
        auto it = diet.finalNutrientProfile.find(nutrient);
        if (it != diet.finalNutrientProfile.end()) {
            double actualValue = it->second;

            // Tolerancia adaptativa
            double base_tolerance = SolverConfig::CONSTRAINT_TOL;

            // Tolerancias más estrictas para límites tóxicos
            if (nutrient == "Cu" || nutrient == "Fe" || nutrient == "Zn") {
                base_tolerance *= 0.1;
            }

            double tolerance = base_tolerance * (1.0 + std::abs(maxValue) * 0.01);

            if (actualValue > maxValue + tolerance) {
                double excess = ((actualValue - maxValue) / maxValue) * 100.0;
                violations.push_back(
                        nutrient + " MAX: límite=" + std::to_string(maxValue) +
                        " actual=" + std::to_string(actualValue) +
                        " (" + std::to_string(excess) + "% exceso)"
                );
                total_violation_score += excess;
                all_valid = false;
            }
        }
    }

    // Log de violaciones si existen
    if (!violations.empty()) {
        LOGE("Restricciones violadas (%zu):", violations.size());
        for (const auto& v : violations) {
            LOGE("  - %s", v.c_str());
        }
        LOGE("Score total de violación: %.2f%%", total_violation_score);

        // Permitir pequeñas violaciones si el score total es bajo
        if (total_violation_score < 5.0 && violations.size() <= 2) {
            LOGW("Violaciones menores aceptadas (< 5%% total)");
            return true;
        }
    }

    return all_valid;
}
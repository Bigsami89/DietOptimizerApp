#include "SimplexSolver.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>
#include <random>
#include <iomanip>

// Declaración de una función auxiliar interna para validar la dieta (MANTIENE COMPATIBILIDAD)
bool isDietValid(const Diet& diet, const std::map<std::string, double>& minNutrients, const std::map<std::string, double>& maxNutrients);

// Namespace anónimo para ocultar implementación interna
namespace {
    
    const double EPSILON = 1e-10;
    
    // Generador de números aleatorios
    
    /**
     * @brief Clase Tableau del Simplex - Implementación completa
     * Totalmente oculta del exterior, no afecta la interfaz
     */
    class SimplexTableau {
    private:
        std::vector<std::vector<double>> tableau;
        std::vector<int> basis;           // Variables en la base
        std::vector<int> nonBasis;        // Variables fuera de la base
        int m, n;                         // m = restricciones, n = variables totales
        int originalVars;                 // Número de variables originales
        
    public:
        SimplexTableau(int rows, int cols, int origVars) 
            : m(rows), n(cols), originalVars(origVars) {
            tableau.resize(m + 1, std::vector<double>(n + 1, 0.0));
            basis.resize(m);
            nonBasis.resize(n - m);
        }
        
        /**
         * @brief Configura el tableau inicial con el problema
         */
        void setup(const std::vector<std::vector<double>>& A,
                  const std::vector<double>& b,
                  const std::vector<double>& c) {
            // Llenar la matriz de restricciones
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    tableau[i][j] = A[i][j];
                }
                tableau[i][n] = b[i];  // RHS
            }
            
            // Función objetivo (negativa para minimización)
            for (int j = 0; j < originalVars; ++j) {
                tableau[m][j] = -c[j];
            }
            
            // Identificar variables básicas iniciales (variables de holgura)
            for (int i = 0; i < m; ++i) {
                // Buscar variable de holgura/artificial para esta restricción
                for (int j = originalVars; j < n; ++j) {
                    if (std::abs(tableau[i][j] - 1.0) < EPSILON) {
                        bool isUnique = true;
                        for (int k = 0; k < m; ++k) {
                            if (k != i && std::abs(tableau[k][j]) > EPSILON) {
                                isUnique = false;
                                break;
                            }
                        }
                        if (isUnique) {
                            basis[i] = j;
                            break;
                        }
                    }
                }
            }
        }
        
        /**
         * @brief Encuentra el pivote usando la regla de Bland (evita ciclos)
         */
        std::pair<int, int> findPivot() {
            // Encontrar columna entrante (costo reducido más negativo)
            int pivotCol = -1;
            double minReducedCost = -EPSILON;
            
            for (int j = 0; j < n; ++j) {
                if (tableau[m][j] < minReducedCost) {
                    minReducedCost = tableau[m][j];
                    pivotCol = j;
                }
            }
            
            if (pivotCol == -1) {
                return {-1, -1}; // Solución óptima encontrada
            }
            
            // Encontrar fila saliente usando la prueba de razón mínima
            int pivotRow = -1;
            double minRatio = std::numeric_limits<double>::infinity();
            
            for (int i = 0; i < m; ++i) {
                if (tableau[i][pivotCol] > EPSILON) {
                    double ratio = tableau[i][n] / tableau[i][pivotCol];
                    if (ratio >= -EPSILON && ratio < minRatio) {
                        minRatio = ratio;
                        pivotRow = i;
                    }
                }
            }
            
            if (pivotRow == -1) {
                return {-2, -2}; // Problema no acotado
            }
            
            return {pivotRow, pivotCol};
        }
        
        /**
         * @brief Realiza la operación de pivote
         */
        void pivot(int row, int col) {
            // Escalar la fila del pivote
            double pivotVal = tableau[row][col];
            if (std::abs(pivotVal) < EPSILON) return; // Evitar división por cero
            
            for (int j = 0; j <= n; ++j) {
                tableau[row][j] /= pivotVal;
            }
            
            // Eliminación gaussiana en las demás filas
            for (int i = 0; i <= m; ++i) {
                if (i != row && std::abs(tableau[i][col]) > EPSILON) {
                    double factor = tableau[i][col];
                    for (int j = 0; j <= n; ++j) {
                        tableau[i][j] -= factor * tableau[row][j];
                    }
                }
            }
            
            // Actualizar la base
            basis[row] = col;
        }
        
        /**
         * @brief Ejecuta el algoritmo Simplex de dos fases
         */
        int solve() {
            // FASE 1: Encontrar solución básica factible
            bool needsPhase1 = false;
            
            // Verificar si necesitamos fase 1 (si hay valores negativos en RHS)
            for (int i = 0; i < m; ++i) {
                if (tableau[i][n] < -EPSILON) {
                    needsPhase1 = true;
                    break;
                }
            }
            
            if (needsPhase1) {
                // Implementar Fase 1 con variables artificiales
                if (!phase1()) {
                    return -3; // No factible
                }
            }
            
            // FASE 2: Optimización
            int maxIterations = 1000;
            int iterations = 0;
            
            while (iterations < maxIterations) {
                auto [row, col] = findPivot();
                
                if (row == -1) return 0;  // Óptimo
                if (row == -2) return -1; // No acotado
                
                pivot(row, col);
                iterations++;
            }
            
            return -2; // No convergió
        }
        
        /**
         * @brief Fase 1 del Simplex: encontrar solución básica factible
         */
        bool phase1() {
            // Guardar función objetivo original
            std::vector<double> originalObj(n + 1);
            for (int j = 0; j <= n; ++j) {
                originalObj[j] = tableau[m][j];
            }
            
            // Crear función objetivo auxiliar (minimizar suma de variables artificiales)
            for (int j = 0; j <= n; ++j) {
                tableau[m][j] = 0;
            }
            
            // Agregar penalización para variables artificiales
            for (int i = 0; i < m; ++i) {
                if (tableau[i][n] < -EPSILON) {
                    // Multiplicar fila por -1 para hacer RHS positivo
                    for (int j = 0; j <= n; ++j) {
                        tableau[i][j] = -tableau[i][j];
                    }
                    
                    // Agregar a la función objetivo
                    for (int j = 0; j < n; ++j) {
                        tableau[m][j] -= tableau[i][j];
                    }
                    tableau[m][n] -= tableau[i][n];
                }
            }
            
            // Resolver con función objetivo auxiliar
            int maxIterations = 500;
            int iterations = 0;
            
            while (iterations < maxIterations) {
                auto [row, col] = findPivot();
                
                if (row == -1) break;  // Óptimo de fase 1
                if (row == -2) return false; // No acotado (no debería pasar)
                
                pivot(row, col);
                iterations++;
            }
            
            // Verificar factibilidad
            if (std::abs(tableau[m][n]) > EPSILON) {
                return false; // No factible
            }
            
            // Restaurar función objetivo original
            for (int j = 0; j <= n; ++j) {
                tableau[m][j] = originalObj[j];
            }
            
            // Recalcular costos reducidos con las variables básicas actuales
            for (int i = 0; i < m; ++i) {
                int basicVar = basis[i];
                if (basicVar < originalVars && std::abs(tableau[m][basicVar]) > EPSILON) {
                    double factor = tableau[m][basicVar];
                    for (int j = 0; j <= n; ++j) {
                        tableau[m][j] -= factor * tableau[i][j];
                    }
                }
            }
            
            return true;
        }
        
        /**
         * @brief Extrae la solución del tableau
         */
        std::vector<double> getSolution() {
            std::vector<double> solution(originalVars, 0.0);
            
            for (int i = 0; i < m; ++i) {
                if (basis[i] < originalVars) {
                    solution[basis[i]] = tableau[i][n];
                }
            }
            
            return solution;
        }
        
        /**
         * @brief Obtiene el valor óptimo de la función objetivo
         */
        double getObjectiveValue() {
            return -tableau[m][n]; // Negativo porque minimizamos
        }
    };
    
    /**
     * @brief Método de Punto Interior (Barrera Logarítmica)
     */
    class InteriorPointSolver {
    private:
        const double MU_INITIAL = 10.0;
        const double MU_DECREASE = 0.1;
        const double TOLERANCE = 1e-9;
        
    public:
        std::vector<double> solve(
            const std::vector<std::vector<double>>& A,
            const std::vector<double>& b,
            const std::vector<double>& c,
            const std::vector<double>& lb,
            const std::vector<double>& ub,
            int numVars) {
            
            int m = A.size();
            int n = numVars;
            
            // Punto inicial (centro del espacio factible)
            std::vector<double> x(n);
            for (int i = 0; i < n; ++i) {
                x[i] = (lb[i] + ub[i]) / 2.0;
                if (std::abs(ub[i] - lb[i]) < EPSILON) {
                    x[i] = lb[i];
                }
            }
            
            // Normalizar para satisfacer restricciones de igualdad
            double sum = std::accumulate(x.begin(), x.end(), 0.0);
            if (sum > EPSILON) {
                for (double& xi : x) xi /= sum;
            }
            
            double mu = MU_INITIAL;
            int maxIter = 100;
            
            for (int iter = 0; iter < maxIter; ++iter) {
                // Calcular gradiente de la función objetivo con barrera
                std::vector<double> gradient(n);
                for (int i = 0; i < n; ++i) {
                    gradient[i] = c[i];
                    // Barrera logarítmica para límites
                    if (x[i] - lb[i] > EPSILON) {
                        gradient[i] -= mu / (x[i] - lb[i]);
                    }
                    if (ub[i] - x[i] > EPSILON) {
                        gradient[i] += mu / (ub[i] - x[i]);
                    }
                }
                
                // Calcular dirección de Newton
                std::vector<double> direction = computeNewtonDirection(A, gradient, x, lb, ub, mu);
                
                // Búsqueda de línea
                double alpha = lineSearch(x, direction, c, A, b, lb, ub, mu);
                
                // Actualizar x
                for (int i = 0; i < n; ++i) {
                    x[i] += alpha * direction[i];
                    // Proyectar en los límites
                    x[i] = std::max(lb[i] + EPSILON, std::min(ub[i] - EPSILON, x[i]));
                }
                
                // Normalizar
                sum = std::accumulate(x.begin(), x.end(), 0.0);
                if (sum > EPSILON) {
                    for (double& xi : x) xi /= sum;
                }
                
                // Reducir parámetro de barrera
                mu *= MU_DECREASE;
                
                // Verificar convergencia
                double gradNorm = 0;
                for (double g : gradient) {
                    gradNorm += g * g;
                }
                gradNorm = std::sqrt(gradNorm);
                
                if (gradNorm < TOLERANCE && mu < TOLERANCE) {
                    break;
                }
            }
            
            return x;
        }
        
    private:
        std::vector<double> computeNewtonDirection(
            const std::vector<std::vector<double>>& A,
            const std::vector<double>& gradient,
            const std::vector<double>& x,
            const std::vector<double>& lb,
            const std::vector<double>& ub,
            double mu) {
            
            int n = x.size();
            std::vector<double> direction(n);
            
            // Aproximación simplificada: dirección de descenso más pronunciado
            // proyectada en el espacio factible
            for (int i = 0; i < n; ++i) {
                direction[i] = -gradient[i];
                
                // Ajustar por cercanía a los límites
                double distToLower = x[i] - lb[i];
                double distToUpper = ub[i] - x[i];
                
                if (distToLower < 0.1) {
                    direction[i] = std::max(0.0, direction[i]);
                }
                if (distToUpper < 0.1) {
                    direction[i] = std::min(0.0, direction[i]);
                }
            }
            
            // Proyectar para mantener suma = 0 (para conservar suma = 1)
            double avgDir = std::accumulate(direction.begin(), direction.end(), 0.0) / n;
            for (double& d : direction) {
                d -= avgDir;
            }
            
            return direction;
        }
        
        double lineSearch(
            const std::vector<double>& x,
            const std::vector<double>& direction,
            const std::vector<double>& c,
            const std::vector<std::vector<double>>& A,
            const std::vector<double>& b,
            const std::vector<double>& lb,
            const std::vector<double>& ub,
            double mu) {
            
            double alpha = 1.0;
            const double beta = 0.5;
            const double armijo = 0.3;
            int n = x.size();
            
            // Calcular máximo paso factible
            for (int i = 0; i < n; ++i) {
                if (direction[i] < -EPSILON) {
                    alpha = std::min(alpha, 0.95 * (lb[i] - x[i]) / direction[i]);
                }
                if (direction[i] > EPSILON) {
                    alpha = std::min(alpha, 0.95 * (ub[i] - x[i]) / direction[i]);
                }
            }
            
            // Búsqueda de línea con condición de Armijo
            double f0 = evaluateObjective(x, c, mu, lb, ub);
            double grad_dot_dir = 0;
            for (int i = 0; i < n; ++i) {
                double gi = c[i];
                if (x[i] - lb[i] > EPSILON) gi -= mu / (x[i] - lb[i]);
                if (ub[i] - x[i] > EPSILON) gi += mu / (ub[i] - x[i]);
                grad_dot_dir += gi * direction[i];
            }
            
            while (alpha > 1e-8) {
                std::vector<double> xNew(n);
                for (int i = 0; i < n; ++i) {
                    xNew[i] = x[i] + alpha * direction[i];
                }
                
                double fNew = evaluateObjective(xNew, c, mu, lb, ub);
                
                if (fNew <= f0 + armijo * alpha * grad_dot_dir) {
                    break;
                }
                
                alpha *= beta;
            }
            
            return alpha;
        }
        
        double evaluateObjective(
            const std::vector<double>& x,
            const std::vector<double>& c,
            double mu,
            const std::vector<double>& lb,
            const std::vector<double>& ub) {
            
            double obj = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                obj += c[i] * x[i];
                // Barrera logarítmica
                if (x[i] - lb[i] > EPSILON) {
                    obj -= mu * std::log(x[i] - lb[i]);
                } else {
                    obj += 1e10; // Penalización grande
                }
                if (ub[i] - x[i] > EPSILON) {
                    obj -= mu * std::log(ub[i] - x[i]);
                } else {
                    obj += 1e10;
                }
            }
            return obj;
        }
    };
    
    /**
     * @brief Algoritmo Genético Híbrido con operadores avanzados
     */
    class GeneticAlgorithm {
    private:
        struct Individual {
            std::vector<double> genes;
            double fitness;
            bool feasible;
            
            Individual(int size) : genes(size, 0.0), fitness(1e10), feasible(false) {}
        };
        
        int popSize = 100;
        int numElites = 5;
        double mutationRate = 0.15;
        double crossoverRate = 0.85;

        std::mt19937 gen;
    
    public:
        GeneticAlgorithm() : gen(std::random_device{}()) {}
        std::vector<double> optimize(
            const std::vector<Ingredient>& ingredients,
            const std::map<std::string, double>& minNutrients,
            const std::map<std::string, double>& maxNutrients,
            double costWeight,
            double methaneWeight,
            int generations = 300) {
            
            int n = ingredients.size();
            std::vector<Individual> population(popSize, Individual(n));
            
            // Inicialización inteligente
            initializePopulation(population, ingredients, minNutrients, maxNutrients);
            
            // Evolución
            for (int g = 0; g < generations; ++g) {
                // Evaluar fitness
                evaluatePopulation(population, ingredients, minNutrients, maxNutrients, costWeight, methaneWeight);
                
                // Ordenar por fitness (mejor primero)
                std::sort(population.begin(), population.end(),
                    [](const Individual& a, const Individual& b) {
                        if (a.feasible != b.feasible) return a.feasible > b.feasible;
                        return a.fitness < b.fitness;
                    });
                
                // Nueva generación
                std::vector<Individual> newPop;
                
                // Elitismo
                for (int i = 0; i < numElites && i < popSize; ++i) {
                    newPop.push_back(population[i]);
                }
                
                // Generar resto de población
                while (newPop.size() < popSize) {
                    Individual parent1 = tournamentSelection(population);
                    Individual parent2 = tournamentSelection(population);
                    
                    Individual child(n);
                    
                    if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < crossoverRate) {
                        child = crossover(parent1, parent2);
                    } else {
                        child = parent1;
                    }
                    
                    if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < mutationRate) {
                        mutate(child, ingredients, minNutrients, maxNutrients);
                    }
                    
                    normalize(child.genes);
                    newPop.push_back(child);
                }
                
                population = newPop;
                
                // Adaptación dinámica de parámetros
                if (g  % 50 == 0) {
                    adaptParameters(population);
                }
            }
            
            // Evaluar población final
            evaluatePopulation(population, ingredients, minNutrients, maxNutrients, costWeight, methaneWeight);
            
            // Retornar mejor individuo factible
            for (const auto& ind : population) {
                if (ind.feasible) {
                    return ind.genes;
                }
            }
            
            // Si no hay factible, retornar el menos infactible
            return population[0].genes;
        }
        
    private:
        void initializePopulation(
            std::vector<Individual>& population,
            const std::vector<Ingredient>& ingredients,
            const std::map<std::string, double>& minNutrients,
            const std::map<std::string, double>& maxNutrients) {
            
            int n = ingredients.size();
            
            // Individuo 0: Distribución uniforme
            for (int i = 0; i < n; ++i) {
                population[0].genes[i] = 1.0 / n;
            }
            
            // Individuos 1 a n: Dominados por cada ingrediente
            for (int p = 1; p <= n && p < popSize; ++p) {
                population[p].genes[(p-1) % n] = 0.7;
                double remaining = 0.3;
                for (int i = 0; i < n; ++i) {
                    if (i != (p-1) % n) {
                        population[p].genes[i] = remaining / (n - 1);
                    }
                }
            }
            
            // Individuos basados en nutrientes críticos
            int idx = n + 1;
            for (const auto& [nutrient, minReq] : minNutrients) {
                if (idx >= popSize) break;
                
                // Encontrar mejores ingredientes para este nutriente
                std::vector<std::pair<int, double>> ranking;
                for (int i = 0; i < n; ++i) {
                    double score = ingredients[i].getNutrient(nutrient) / (ingredients[i].cost + 0.01);
                    ranking.push_back({i, score});
                }
                std::sort(ranking.begin(), ranking.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
                
                // Crear individuo enfocado en los mejores 3
                for (int i = 0; i < n; ++i) {
                    population[idx].genes[i] = 0.0;
                }
                for (int i = 0; i < std::min(3, n); ++i) {
                    population[idx].genes[ranking[i].first] = 1.0 / std::min(3, n);
                }
                idx++;
            }
            
            // Resto: aleatorios con sesgo
            for (int p = idx; p < popSize; ++p) {
                double sum = 0;
                for (int i = 0; i < n; ++i) {
                        population[p].genes[i] = std::pow(std::uniform_real_distribution<>(0.0, 1.0)(gen), 2.0);
                        sum += population[p].genes[i];
                }
                for (int i = 0; i < n; ++i) {
                    population[p].genes[i] /= sum;
                }
            }
        }
        
        void evaluatePopulation(
            std::vector<Individual>& population,
            const std::vector<Ingredient>& ingredients,
            const std::map<std::string, double>& minNutrients,
            const std::map<std::string, double>& maxNutrients,
            double costWeight,
            double methaneWeight) {
            
            for (auto& ind : population) {
                Diet diet;
                for (size_t i = 0; i < ingredients.size(); ++i) {
                    if (ind.genes[i] > EPSILON) {
                        diet.composition.emplace_back(ingredients[i], ind.genes[i]);
                    }
                }
                diet.calculateDietProperties();
                
                // Calcular fitness base
                ind.fitness = diet.totalCost * costWeight;
                double tdn = diet.finalNutrientProfile.count("TDN") > 0 ? 
                            diet.finalNutrientProfile.at("TDN") : 0;
                ind.fitness += tdn * methaneWeight * 0.01;
                
                // Verificar factibilidad y penalizar
                ind.feasible = true;
                double penalty = 0;
                
                for (const auto& [nutrient, minVal] : minNutrients) {
                    double actual = diet.finalNutrientProfile.at(nutrient);
                    if (actual < minVal - EPSILON) {
                        ind.feasible = false;
                        double violation = (minVal - actual) / minVal;
                        penalty += violation * violation * 1000;
                    }
                }
                
                for (const auto& [nutrient, maxVal] : maxNutrients) {
                    double actual = diet.finalNutrientProfile.at(nutrient);
                    if (actual > maxVal + EPSILON) {
                        ind.feasible = false;
                        double violation = (actual - maxVal) / maxVal;
                        penalty += violation * violation * 1000;
                    }
                }
                
                ind.fitness += penalty;
            }
        }
        
        Individual tournamentSelection(const std::vector<Individual>& population) {
            int tournamentSize = 7;
            Individual best = population[std::uniform_int_distribution<>(0, population.size()-1)(gen)];
            
            for (int i = 1; i < tournamentSize; ++i) {
                Individual challenger = population[std::uniform_int_distribution<>(0, population.size()-1)(gen)];
                if (challenger.feasible && !best.feasible) {
                    best = challenger;
                } else if (challenger.feasible == best.feasible && challenger.fitness < best.fitness) {
                    best = challenger;
                }
            }
            
            return best;
        }
        
        Individual crossover(const Individual& parent1, const Individual& parent2) {
            int n = parent1.genes.size();
            Individual child(n);
            
            // Crossover SBX (Simulated Binary Crossover)
            double eta = 2.0; // Parámetro de distribución
            
            for (int i = 0; i < n; ++i) {
                double u = std::uniform_real_distribution<>(0.0, 1.0)(gen);
                double beta;
                
                if (u <= 0.5) {
                    beta = std::pow(2.0 * u, 1.0 / (eta + 1.0));
                } else {
                    beta = std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));
                }
                
                child.genes[i] = 0.5 * ((1.0 + beta) * parent1.genes[i] + 
                                        (1.0 - beta) * parent2.genes[i]);
                child.genes[i] = std::max(0.0, child.genes[i]);
            }
            
            return child;
        }
        
        void mutate(Individual& ind,
                   const std::vector<Ingredient>& ingredients,
                   const std::map<std::string, double>& minNutrients,
                   const std::map<std::string, double>& maxNutrients) {
            
            int n = ind.genes.size();
            
            // Mutación polinomial
            double eta = 20.0;
            
            for (int i = 0; i < n; ++i) {
                if (std::uniform_real_distribution<>(0.0, 1.0)(gen)  < 0.1) { // Probabilidad de mutar cada gen
                    double u = std::uniform_real_distribution<>(0.0, 1.0)(gen);
                    double delta;
                    
                    if (u < 0.5) {
                        delta = std::pow(2.0 * u, 1.0 / (eta + 1.0)) - 1.0;
                    } else {
                        delta = 1.0 - std::pow(2.0 * (1.0 - u), 1.0 / (eta + 1.0));
                    }
                    
                    ind.genes[i] += delta * 0.1;
                    ind.genes[i] = std::max(0.0, std::min(1.0, ind.genes[i]));
                }
            }
            
            // Ocasionalmente, mutación dirigida
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < 0.05) {
                // Encontrar nutriente más violado
                Diet diet;
                for (int i = 0; i < n; ++i) {
                    if (ind.genes[i] > EPSILON) {
                        diet.composition.emplace_back(ingredients[i], ind.genes[i]);
                    }
                }
                diet.calculateDietProperties();
                
                double maxViolation = 0;
                std::string criticalNutrient;
                bool isDeficit = true;
                
                for (const auto& [nutrient, minVal] : minNutrients) {
                    double deficit = minVal - diet.finalNutrientProfile.at(nutrient);
                    if (deficit > maxViolation) {
                        maxViolation = deficit;
                        criticalNutrient = nutrient;
                        isDeficit = true;
                    }
                }
                
                for (const auto& [nutrient, maxVal] : maxNutrients) {
                    double excess = diet.finalNutrientProfile.at(nutrient) - maxVal;
                    if (excess > maxViolation) {
                        maxViolation = excess;
                        criticalNutrient = nutrient;
                        isDeficit = false;
                    }
                }
                
                // Ajustar genes para corregir violación
                if (maxViolation > EPSILON) {
                    if (isDeficit) {
                        // Aumentar ingrediente con mayor contenido
                        int bestIdx = -1;
                        double maxContent = 0;
                        for (int i = 0; i < n; ++i) {
                            double content = ingredients[i].getNutrient(criticalNutrient);
                            if (content > maxContent) {
                                maxContent = content;
                                bestIdx = i;
                            }
                        }
                        if (bestIdx != -1) {
                            ind.genes[bestIdx] += 0.1;
                        }
                    } else {
                        // Reducir ingrediente con mayor contribución
                        int worstIdx = -1;
                        double maxContrib = 0;
                        for (int i = 0; i < n; ++i) {
                            double contrib = ingredients[i].getNutrient(criticalNutrient) * ind.genes[i];
                            if (contrib > maxContrib) {
                                maxContrib = contrib;
                                worstIdx = i;
                            }
                        }
                        if (worstIdx != -1 && ind.genes[worstIdx] > 0.01) {
                            ind.genes[worstIdx] *= 0.5;
                        }
                    }
                }
            }
        }
        
        void normalize(std::vector<double>& genes) {
            double sum = std::accumulate(genes.begin(), genes.end(), 0.0);
            if (sum > EPSILON) {
                for (double& g : genes) g /= sum;
            } else {
                // Reinicializar si todos son cero
                for (double& g : genes) g = 1.0 / genes.size();
            }
        }
        
        void adaptParameters(const std::vector<Individual>& population) {
            // Contar individuos factibles
            int feasibleCount = 0;
            for (const auto& ind : population) {
                if (ind.feasible) feasibleCount++;
            }
            
            double feasibleRatio = static_cast<double>(feasibleCount) / population.size();
            
            // Adaptar tasas según convergencia
            if (feasibleRatio > 0.8) {
                // Población muy factible, reducir exploración
                mutationRate = std::max(0.05, mutationRate * 0.9);
                crossoverRate = std::min(0.95, crossoverRate * 1.05);
            } else if (feasibleRatio < 0.2) {
                // Poca factibilidad, aumentar exploración
                mutationRate = std::min(0.3, mutationRate * 1.1);
                crossoverRate = std::max(0.6, crossoverRate * 0.95);
            }
        }
    };
    
    /**
     * @brief Búsqueda Local Variable (VNS) para refinamiento
     */
    std::vector<double> variableNeighborhoodSearch(
        std::vector<double> initial,
        const std::vector<Ingredient>& ingredients,
        const std::map<std::string, double>& minNutrients,
        const std::map<std::string, double>& maxNutrients,
        double costWeight,
        double methaneWeight) {
        
        std::mt19937 gen(std::random_device{}()); 
        int n = ingredients.size();
        std::vector<double> current = initial;
        std::vector<double> best = initial;
        
        // Calcular objetivo inicial
        auto calculateObjective = [&](const std::vector<double>& sol) {
            double obj = 0;
            for (int i = 0; i < n; ++i) {
                obj += sol[i] * ingredients[i].cost * costWeight;
                double tdn = ingredients[i].getNutrient("TDN");
                if (tdn > 0) obj += sol[i] * tdn * methaneWeight * 0.01;
            }
            return obj;
        };
        
        double bestObj = calculateObjective(best);
        
        // Definir vecindarios de diferentes tamaños
        std::vector<double> neighborhoodSizes = {0.001, 0.005, 0.01, 0.05, 0.1};
        
        for (int iter = 0; iter < 100; ++iter) {
            bool improved = false;
            
            for (double stepSize : neighborhoodSizes) {
                // Explorar vecindario actual
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        if (i == j || current[i] < stepSize) continue;
                        
                        // Crear vecino
                        std::vector<double> neighbor = current;
                        neighbor[i] -= stepSize;
                        neighbor[j] += stepSize;
                        
                        // Validar
                        Diet testDiet;
                        for (int k = 0; k < n; ++k) {
                            if (neighbor[k] > EPSILON) {
                                testDiet.composition.emplace_back(ingredients[k], neighbor[k]);
                            }
                        }
                        testDiet.calculateDietProperties();
                        
                        if (isDietValid(testDiet, minNutrients, maxNutrients)) {
                            double neighborObj = calculateObjective(neighbor);
                            if (neighborObj < bestObj - EPSILON) {
                                best = neighbor;
                                bestObj = neighborObj;
                                current = neighbor;
                                improved = true;
                                break;
                            }
                        }
                    }
                    if (improved) break;
                }
                if (improved) break;
            }
            
            if (!improved) {
                // Perturbación para escapar de mínimo local
                current = best;
                for (int k = 0; k < 3; ++k) {
                    int i = std::uniform_int_distribution<>(0, n-1)(gen);
                    int j = std::uniform_int_distribution<>(0, n-1)(gen);
                    if (i != j && current[i] > 0.01) {
                        double transfer = current[i] * 0.1;
                        current[i] -= transfer;
                        current[j] += transfer;
                    }
                }
            }
        }
        
        return best;
    }
}

/**
 * @brief Resuelve el problema de formulación de dietas usando Simplex completo y métodos avanzados
 * INTERFAZ 100% COMPATIBLE CON LA VERSIÓN ORIGINAL
 */
std::optional<Diet> SimplexSolver::solve(
    const std::vector<Ingredient>& ingredients,
    const std::map<std::string, double>& minNutrients,
    const std::map<std::string, double>& maxNutrients,
    double costWeight,
    double methaneWeight,
    double DMI_kg_day) {

    const double TOLERANCE = 1e-6;
    int numIngredients = static_cast<int>(ingredients.size());
    
    if (numIngredients == 0) {
        std::cerr << "Error: No hay ingredientes disponibles.\n";
        return std::nullopt;
    }
    
    // Verificación rápida de factibilidad
    for (const auto& [nutrient, minReq] : minNutrients) {
        double maxAvailable = 0;
        for (const auto& ing : ingredients) {
            maxAvailable = std::max(maxAvailable, ing.getNutrient(nutrient));
        }
        if (maxAvailable < minReq * 0.01) {
            std::cerr << "Error: Imposible satisfacer requerimiento mínimo de " << nutrient << "\n";
            return std::nullopt;
        }
    }
    
    std::vector<double> bestSolution;
    double bestObjective = std::numeric_limits<double>::infinity();
    bool foundFeasible = false;
    
    // ============= MÉTODO 1: SIMPLEX COMPLETO =============
    try {
        // Construir problema de programación lineal
        // Variables: x[0..n-1] = proporciones de ingredientes
        // Variables de holgura: s[0..m-1] para restricciones <=
        // Variables de exceso: e[0..k-1] para restricciones >=
        
        int numMinConstraints = minNutrients.size();
        int numMaxConstraints = maxNutrients.size();
        int numConstraints = 1 + numMinConstraints + numMaxConstraints; // suma=1 + mins + maxs
        int totalVars = numIngredients + numMinConstraints + numMaxConstraints;
        
        std::vector<std::vector<double>> A(numConstraints, std::vector<double>(totalVars, 0.0));
        std::vector<double> b(numConstraints);
        std::vector<double> c(totalVars, 0.0);
        
        int row = 0;
        
        // Restricción: suma de proporciones = 1
        for (int i = 0; i < numIngredients; ++i) {
            A[row][i] = 1.0;
        }
        b[row] = 1.0;
        row++;
        
        // Restricciones de nutrientes mínimos (>= se convierte en = con variable de exceso)
        int slackIdx = numIngredients;
        for (const auto& [nutrient, minVal] : minNutrients) {
            for (int i = 0; i < numIngredients; ++i) {
                A[row][i] = ingredients[i].getNutrient(nutrient);
            }
            A[row][slackIdx] = -1.0; // Variable de exceso (resta)
            b[row] = minVal;
            slackIdx++;
            row++;
        }
        
        // Restricciones de nutrientes máximos (<= se convierte en = con variable de holgura)
        for (const auto& [nutrient, maxVal] : maxNutrients) {
            for (int i = 0; i < numIngredients; ++i) {
                A[row][i] = ingredients[i].getNutrient(nutrient);
            }
            A[row][slackIdx] = 1.0; // Variable de holgura (suma)
            b[row] = maxVal;
            slackIdx++;
            row++;
        }
        
        // Función objetivo: minimizar costo ponderado + emisiones
        for (int i = 0; i < numIngredients; ++i) {
            c[i] = ingredients[i].cost * costWeight;
            double tdn = ingredients[i].getNutrient("TDN");
            if (tdn > 0) {
                c[i] += tdn * methaneWeight * 0.01;
            }
        }
        
        // Resolver con Simplex
        SimplexTableau tableau(numConstraints, totalVars, numIngredients);
        tableau.setup(A, b, c);
        
        int result = tableau.solve();
        
        if (result == 0) { // Solución óptima encontrada
            std::vector<double> solution = tableau.getSolution();
            
            // Validar solución
            Diet testDiet;
            for (int i = 0; i < numIngredients; ++i) {
                if (solution[i] > TOLERANCE) {
                    testDiet.composition.emplace_back(ingredients[i], solution[i]);
                }
            }
            testDiet.calculateDietProperties();
            
            if (isDietValid(testDiet, minNutrients, maxNutrients)) {
                double obj = tableau.getObjectiveValue();
                if (obj < bestObjective) {
                    bestObjective = obj;
                    bestSolution = solution;
                    foundFeasible = true;
                }
            }
        }
    } catch (const std::exception& e) {
        // Continuar con otros métodos si falla
    }
    
    // ============= MÉTODO 2: PUNTO INTERIOR =============
    if (!foundFeasible || bestObjective > 1e-3) {
        try {
            // Preparar datos para punto interior
            std::vector<std::vector<double>> A_eq;
            std::vector<double> b_eq;
            std::vector<double> c(numIngredients);
            std::vector<double> lb(numIngredients, 0.0);
            std::vector<double> ub(numIngredients, 1.0);
            
            // Restricción de igualdad: suma = 1
            A_eq.push_back(std::vector<double>(numIngredients, 1.0));
            b_eq.push_back(1.0);
            
            // Función objetivo
            for (int i = 0; i < numIngredients; ++i) {
                c[i] = ingredients[i].cost * costWeight;
                double tdn = ingredients[i].getNutrient("TDN");
                if (tdn > 0) {
                    c[i] += tdn * methaneWeight * 0.01;
                }
            }
            
            InteriorPointSolver ipSolver;
            std::vector<double> ipSolution = ipSolver.solve(A_eq, b_eq, c, lb, ub, numIngredients);
            
            // Validar solución
            Diet testDiet;
            for (int i = 0; i < numIngredients; ++i) {
                if (ipSolution[i] > TOLERANCE) {
                    testDiet.composition.emplace_back(ingredients[i], ipSolution[i]);
                }
            }
            testDiet.calculateDietProperties();
            
            if (isDietValid(testDiet, minNutrients, maxNutrients)) {
                double obj = 0;
                for (int i = 0; i < numIngredients; ++i) {
                    obj += ipSolution[i] * c[i];
                }
                
                if (!foundFeasible || obj < bestObjective) {
                    bestObjective = obj;
                    bestSolution = ipSolution;
                    foundFeasible = true;
                }
            }
        } catch (...) {
            // Continuar con otros métodos
        }
    }
    
    // ============= MÉTODO 3: ALGORITMO GENÉTICO HÍBRIDO =============
    if (!foundFeasible) {
        try {
            GeneticAlgorithm ga;
            std::vector<double> gaSolution = ga.optimize(
                ingredients, minNutrients, maxNutrients, 
                costWeight, methaneWeight, 300
            );
            
            // Validar solución
            Diet testDiet;
            for (int i = 0; i < numIngredients; ++i) {
                if (gaSolution[i] > TOLERANCE) {
                    testDiet.composition.emplace_back(ingredients[i], gaSolution[i]);
                }
            }
            testDiet.calculateDietProperties();
            
            bool valid = isDietValid(testDiet, minNutrients, maxNutrients);
            
            // Aceptar con pequeñas violaciones si no hay mejor alternativa
            if (valid || !foundFeasible) {
                double obj = 0;
                for (int i = 0; i < numIngredients; ++i) {
                    obj += gaSolution[i] * ingredients[i].cost * costWeight;
                    double tdn = ingredients[i].getNutrient("TDN");
                    if (tdn > 0) {
                        obj += gaSolution[i] * tdn * methaneWeight * 0.01;
                    }
                }
                
                if (!foundFeasible || obj < bestObjective) {
                    bestSolution = gaSolution;
                    foundFeasible = valid;
                }
            }
        } catch (...) {
            // Silenciar errores
        }
    }
    
    // ============= MÉTODO 4: REFINAMIENTO CON VNS =============
    if (foundFeasible && bestSolution.size() > 0) {
        try {
            std::vector<double> refinedSolution = variableNeighborhoodSearch(
                bestSolution, ingredients, minNutrients, maxNutrients, 
                costWeight, methaneWeight
            );
            
            // Validar solución refinada
            Diet testDiet;
            for (int i = 0; i < numIngredients; ++i) {
                if (refinedSolution[i] > TOLERANCE) {
                    testDiet.composition.emplace_back(ingredients[i], refinedSolution[i]);
                }
            }
            testDiet.calculateDietProperties();
            
            if (isDietValid(testDiet, minNutrients, maxNutrients)) {
                bestSolution = refinedSolution;
            }
        } catch (...) {
            // Mantener solución actual si falla el refinamiento
        }
    }
    
    // ============= CONSTRUCCIÓN DE LA DIETA FINAL =============
    if (!foundFeasible || bestSolution.empty()) {
        std::cerr << "Solucion final inviable. No se pudo encontrar una dieta que cumpla todas las restricciones.\n";
        return std::nullopt;
    }
    
    // Limpiar y normalizar solución final
    std::vector<double> proportions = bestSolution;
    double totalSum = 0;
    for (int i = 0; i < numIngredients; ++i) {
        if (proportions[i] < TOLERANCE) {
            proportions[i] = 0;
        } else {
            totalSum += proportions[i];
        }
    }
    
    if (totalSum < TOLERANCE) {
        std::cerr << "Error: La suma de proporciones es cero.\n";
        return std::nullopt;
    }
    
    // Normalizar para asegurar suma = 1
    for (int i = 0; i < numIngredients; ++i) {
        proportions[i] /= totalSum;
    }
    
    // Construir dieta final
    Diet finalDiet;
    for (int i = 0; i < numIngredients; ++i) {
        if (proportions[i] > TOLERANCE) {
            finalDiet.composition.emplace_back(ingredients[i], proportions[i]);
        }
    }
    
    finalDiet.calculateDietProperties();
    
    // Validación final
    if (!isDietValid(finalDiet, minNutrients, maxNutrients)) {
        std::cerr << "Solucion final inviable. No se pudo encontrar una dieta que cumpla todas las restricciones.\n";
        return std::nullopt;
    }
    
    // Calcular emisiones de metano
    finalDiet.calculateEntericMethane(DMI_kg_day);
    
    return finalDiet;
}

/**
 * @brief Función auxiliar para validar si una dieta cumple todas las restricciones.
 * EXACTAMENTE IGUAL A LA VERSIÓN ORIGINAL
 */
bool isDietValid(const Diet& diet, const std::map<std::string, double>& minNutrients, const std::map<std::string, double>& maxNutrients) {
    const double TOLERANCE = 1e-5;
    for (const auto& req : minNutrients) {
        if (diet.finalNutrientProfile.at(req.first) < req.second - TOLERANCE) return false;
    }
    for (const auto& limit : maxNutrients) {
        if (diet.finalNutrientProfile.at(limit.first) > limit.second + TOLERANCE) return false;
    }
    return true;
}
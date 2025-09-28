#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h> // <-- LIBRERÍA DE LOGGING DE ANDROID
#include "DietApi.h"

// Define un tag para filtrar los mensajes en Logcat
#define LOG_TAG "DietCalculatorCPP"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jobject JNICALL
Java_com_example_dietoptimizerapp_jni_JniBridge_calculateDiets(
        JNIEnv *env,
        jobject /* this */,
        jstring jsonInput) {

    // --- LOG DE DEPURACIÓN 1 ---
    LOGI("Función C++ 'calculateDiets' llamada.");

    const char *json_c_str = env->GetStringUTFChars(jsonInput, nullptr);
    if (json_c_str == nullptr) {
        LOGE("Error: GetStringUTFChars devolvió null.");
        return nullptr;
    }
    std::string json_std_str(json_c_str);
    env->ReleaseStringUTFChars(jsonInput, json_c_str);

    // --- LOG DE DEPURACIÓN 2 ---
    LOGI("JSON recibido del lado nativo: %s", json_std_str.substr(0, 200).c_str()); // Imprime los primeros 200 caracteres

    C_DietResult *c_results = nullptr;
    int results_count = 0;
    int error_code = calculate_diets_from_json(json_std_str.c_str(), &c_results, &results_count);

    if (error_code != 0 || results_count == 0) {
        // --- LOG DE DEPURACIÓN 3 (Error) ---
        LOGE("El cálculo falló o no devolvió resultados. Código de error: %d", error_code);
        if (c_results) free_diet_results(c_results, results_count);
        return nullptr;
    }

    // --- LOG DE DEPURACIÓN 4 (Éxito) ---
    LOGI("Cálculo exitoso. Se encontraron %d dietas. Convirtiendo a objetos de Java...", results_count);

    // ... (El resto del código para convertir los resultados a objetos de Java no cambia) ...

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListCtor = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jclass dietResultClass = env->FindClass("com/example/dietoptimizerapp/models/DietResult");
    jmethodID dietResultCtor = env->GetMethodID(dietResultClass, "<init>", "()V");
    jfieldID totalCostField = env->GetFieldID(dietResultClass, "totalCost", "D");
    jfieldID totalMethaneField = env->GetFieldID(dietResultClass, "totalMethane", "D");
    jfieldID componentsField = env->GetFieldID(dietResultClass, "components", "Ljava/util/ArrayList;");

    jclass componentClass = env->FindClass("com/example/dietoptimizerapp/models/DietComponent");
    jmethodID componentCtor = env->GetMethodID(componentClass, "<init>", "(Ljava/lang/String;D)V");

    jobject resultsList = env->NewObject(arrayListClass, arrayListCtor);

    for (int i = 0; i < results_count; ++i) {
        jobject dietResultObj = env->NewObject(dietResultClass, dietResultCtor);
        env->SetDoubleField(dietResultObj, totalCostField, c_results[i].totalCost);
        env->SetDoubleField(dietResultObj, totalMethaneField, c_results[i].totalMethane);

        jobject componentsList = env->NewObject(arrayListClass, arrayListCtor);
        for (int j = 0; j < c_results[i].componentsCount; ++j) {
            jstring ingredientName = env->NewStringUTF(c_results[i].components[j].ingredientName);
            double proportion = c_results[i].components[j].proportion;
            jobject componentObj = env->NewObject(componentClass, componentCtor, ingredientName, proportion);
            env->CallBooleanMethod(componentsList, arrayListAdd, componentObj);
            env->DeleteLocalRef(ingredientName);
            env->DeleteLocalRef(componentObj);
        }
        env->SetObjectField(dietResultObj, componentsField, componentsList);

        env->CallBooleanMethod(resultsList, arrayListAdd, dietResultObj);
        env->DeleteLocalRef(dietResultObj);
        env->DeleteLocalRef(componentsList);
    }

    free_diet_results(c_results, results_count);
    LOGI("Conversión a Java finalizada. Devolviendo resultados.");
    return resultsList;
}
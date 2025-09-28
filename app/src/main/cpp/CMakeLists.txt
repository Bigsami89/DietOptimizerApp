cmake_minimum_required(VERSION 3.22.1)

# Nombre del proyecto
project("dietoptimizer")

# Configurar estándar C++17 (requerido para tu código)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuraciones del compilador
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Configuraciones específicas por tipo de build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    message(STATUS "Build type: Release with optimizations")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
    message(STATUS "Build type: Debug")
endif()

# Directorios de includes
include_directories(include)

# Encontrar librerías del sistema Android requeridas
find_library(log-lib log)
find_library(android-lib android)

# Verificar que las librerías se encontraron
if(NOT log-lib)
    message(FATAL_ERROR "log library not found")
endif()

if(NOT android-lib)
    message(FATAL_ERROR "android library not found")
endif()

# Archivos fuente de tu librería C++
set(SOURCES
        # Implementaciones principales
        src/ApiRequest.cpp
        src/DietApi.cpp
        src/Ingredient.cpp
        src/RationCalculator.cpp
        src/SimplexSolver.cpp

        # Bridge JNI
        jni_bridge.cpp
)

# Verificar que todos los archivos fuente existen
foreach(SOURCE_FILE ${SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        message(WARNING "Source file ${SOURCE_FILE} not found")
    endif()
endforeach()

# Crear la librería compartida principal
add_library(dietoptimizer SHARED ${SOURCES})

# Configurar propiedades específicas de la librería
set_target_properties(dietoptimizer PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# Enlazar con las librerías del sistema
target_link_libraries(dietoptimizer
        ${log-lib}
        ${android-lib}
)

# Definiciones del compilador
target_compile_definitions(dietoptimizer PRIVATE
        ANDROID=1
        API_EXPORT_DEFINITION=1
        __ANDROID_API__=${ANDROID_NATIVE_API_LEVEL}
)

# Configuraciones específicas por arquitectura
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(dietoptimizer PRIVATE -march=armv8-a)
    message(STATUS "Compiling for ARM64")
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    target_compile_options(dietoptimizer PRIVATE -march=armv7-a -mfpu=neon)
    message(STATUS "Compiling for ARMv7")
elseif(ANDROID_ABI STREQUAL "x86")
    target_compile_options(dietoptimizer PRIVATE -march=i686 -mssse3 -mfpmath=sse -m32)
    message(STATUS "Compiling for x86")
elseif(ANDROID_ABI STREQUAL "x86_64")
    target_compile_options(dietoptimizer PRIVATE -march=x86-64 -msse4.2 -mpopcnt -m64 -mfpmath=sse)
    message(STATUS "Compiling for x86_64")
endif()

# Configuraciones de optimización adicionales para Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dietoptimizer PRIVATE
            -ffast-math
            -fno-math-errno
            -funsafe-math-optimizations
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -freciprocal-math
    )

    # Link Time Optimization (LTO) si está disponible
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set_property(TARGET dietoptimizer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(STATUS "LTO not supported: ${output}")
    endif()
endif()

# Información de debug para desarrollo
message(STATUS "=== Build Configuration ===")
message(STATUS "Android ABI: ${ANDROID_ABI}")
message(STATUS "Android API Level: ${ANDROID_NATIVE_API_LEVEL}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Source Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler Flags: ${CMAKE_CXX_FLAGS}")

# Imprimir archivos fuente encontrados
message(STATUS "Source Files:")
foreach(SOURCE_FILE ${SOURCES})
    message(STATUS "  - ${SOURCE_FILE}")
endforeach()

# Configuración de símbolos para debugging en Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Mantener algunos símbolos para crash reporting
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1")
endif()

# Configurar warnings específicos
target_compile_options(dietoptimizer PRIVATE
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-sign-compare
        -Wno-missing-field-initializers
)

# Configurar includes del sistema (opcional)
target_include_directories(dietoptimizer SYSTEM PRIVATE
        ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/include
)

message(STATUS "=== Configuration Complete ===")

# Para debugging: imprimir todas las variables de CMake (descomenta si necesitas debug)
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
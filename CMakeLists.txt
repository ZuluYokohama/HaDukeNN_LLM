cmake_minimum_required(VERSION 3.20)
project(hadukenn_tri_machine_hal LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HADUKENN_TARGET_MACHINE "Target machine profile: auto|wbeast|lapbeast|zenbook" "auto")
option(HADUKENN_ENABLE_CUDA "Build Blackwell sm_120 and Turing sm_75 CUDA kernels when nvcc is present" ON)
option(HADUKENN_BUILD_FAST_HEAD_NATIVE "Build optional standalone fast_head_native shared library" ON)

find_package(OpenMP QUIET)
find_package(Threads REQUIRED)
include(CheckCCompilerFlag)

add_library(hadukenn_tri_hal SHARED
    src/hadukenn_core/csrc/tri_machine_hal.c
    src/hadukenn_core/csrc/fast_head_avx512.c
)
target_include_directories(hadukenn_tri_hal PUBLIC src/hadukenn_core/csrc)

if(OpenMP_C_FOUND)
    target_link_libraries(hadukenn_tri_hal PUBLIC OpenMP::OpenMP_C)
else()
    target_link_libraries(hadukenn_tri_hal PUBLIC Threads::Threads)
endif()

if(UNIX)
    target_link_libraries(hadukenn_tri_hal PUBLIC m)
endif()

set(HADUKENN_ARM64_ISA_FLAG "")
if(NOT MSVC AND (HADUKENN_TARGET_MACHINE STREQUAL "zenbook" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    foreach(_cand IN ITEMS
        "-mcpu=oryon-1+fp16"
        "-mcpu=oryon-1"
        "-march=armv8.6-a+dotprod+i8mm+fp16"
        "-march=armv8.2-a+dotprod"
        "-march=native"
    )
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" _var "HADUKENN_C_FLAG_${_cand}")
        check_c_compiler_flag("${_cand}" ${_var})
        if(${_var})
            set(HADUKENN_ARM64_ISA_FLAG "${_cand}")
            break()
        endif()
    endforeach()
endif()

# Machine-specific lowest-level ISA flags
if(MSVC)
    target_compile_options(hadukenn_tri_hal PRIVATE /O2 /fp:fast)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        message(STATUS "Configuring MSVC ARM64 build for Zenbook (Snapdragon X Oryon)")
    else()
        message(STATUS "Configuring MSVC x86_64 AVX2+FMA3 build for WBeast / LapBeast")
        target_compile_options(hadukenn_tri_hal PRIVATE /arch:AVX2)
    endif()
else()
    if(HADUKENN_TARGET_MACHINE STREQUAL "zenbook" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        message(STATUS "Configuring GCC/Clang ARM64 NEON+SDOT+I8MM build for Zenbook (${HADUKENN_ARM64_ISA_FLAG})")
        if(HADUKENN_ARM64_ISA_FLAG)
            target_compile_options(hadukenn_tri_hal PRIVATE -O3 ${HADUKENN_ARM64_ISA_FLAG})
        else()
            target_compile_options(hadukenn_tri_hal PRIVATE -O3)
        endif()
    elseif(HADUKENN_TARGET_MACHINE STREQUAL "wbeast")
        message(STATUS "Configuring GCC/Clang x86_64 AVX2+FMA3+BMI2 build for WBeast (i9-9900K)")
        target_compile_options(hadukenn_tri_hal PRIVATE -O3 -mavx2 -mfma -mbmi2)
    elseif(HADUKENN_TARGET_MACHINE STREQUAL "lapbeast")
        message(STATUS "Configuring GCC/Clang x86_64 AVX2+FMA3+AVX-VNNI build for LapBeast (275HX)")
        target_compile_options(hadukenn_tri_hal PRIVATE -O3 -mavx2 -mfma -mavxvnni)
    else()
        message(STATUS "Configuring native host-tuned build (-O3 -march=native)")
        target_compile_options(hadukenn_tri_hal PRIVATE -O3 -march=native)
    endif()
endif()

if(HADUKENN_BUILD_FAST_HEAD_NATIVE AND UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/hadukenn_core/csrc/fast_head_native.c")
    add_library(hadukenn_fast_head_native SHARED
        src/hadukenn_core/csrc/fast_head_native.c
    )
    target_link_libraries(hadukenn_fast_head_native PUBLIC Threads::Threads m)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        if(HADUKENN_ARM64_ISA_FLAG)
            target_compile_options(hadukenn_fast_head_native PRIVATE -O3 ${HADUKENN_ARM64_ISA_FLAG})
        else()
            target_compile_options(hadukenn_fast_head_native PRIVATE -O3)
        endif()
    else()
        target_compile_options(hadukenn_fast_head_native PRIVATE -O3 -march=native)
    endif()
endif()

if(HADUKENN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        add_library(hadukenn_cuda_kernels SHARED
            src/hadukenn_core/csrc/cuda_blackwell_turing_kernels.cu
        )
        if(HADUKENN_TARGET_MACHINE STREQUAL "wbeast")
            # Dual-GPU WBeast: RTX 5070 (sm_120) + GTX 1660 Ti (sm_75)
            set(CMAKE_CUDA_ARCHITECTURES "75;120")
            set_target_properties(hadukenn_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;120")
        else()
            # LapBeast RTX 5060 Laptop (sm_120)
            set_target_properties(hadukenn_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "120")
        endif()
        target_compile_options(hadukenn_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    endif()
endif()

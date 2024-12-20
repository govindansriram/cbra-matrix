cmake_minimum_required(VERSION 3.22)
project(ConBasedFilter VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(SOURCES
        include/matrix.h
        src/allocator.h
        src/math_dispatcher.h
        src/standard_kernal/standard_allocator.h
        src/standard_kernal/standard_allocator.cpp
        src/allocator.cpp
        src/standard_kernal/math.h
        src/matrix.cpp
        src/enums.cpp
        include/enums.h
)

list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/latest/lib/cmake") # path to intel onetbb

add_library(CmlContentBasedFiltering SHARED ${SOURCES})

# Include directories for public headers
target_include_directories(CmlContentBasedFiltering PUBLIC include)

target_include_directories(CmlContentBasedFiltering PRIVATE src/standard_kernal)

# Define install rules
install(TARGETS CmlContentBasedFiltering
        DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include
)

find_package(TBB REQUIRED)

target_link_libraries(CmlContentBasedFiltering PUBLIC TBB::tbb)

## Specify the include directory where your header files are located

option(ENABLE_TESTING "Enable testing-specific compile options" ON)
option(IS_NAIVE "Enable testing-specific compile options" OFF)

if(ENABLE_TESTING)

    set(COMMON_COMPILE_OPTIONS
            -ggdb
            -pedantic-errors
            -O0
            -Wall
            -Weffc++
            -Wextra
            -Wconversion
            -Wsign-conversion
            -Werror
    )

    target_compile_options(CmlContentBasedFiltering PRIVATE ${COMMON_COMPILE_OPTIONS})

    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
    )

    # Make googletest available (this will download and build it)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_matrix tests/test_matrix.cpp)
    include(GoogleTest)
    gtest_discover_tests(test_matrix)

    target_compile_options(test_matrix PRIVATE ${COMMON_COMPILE_OPTIONS})

    target_link_libraries(
            test_matrix
            GTest::gtest_main
            CmlContentBasedFiltering
            TBB::tbb
    )

else ()

    find_package(benchmark REQUIRED)

    set(COMMON_COMPILE_OPTIONS
            -O3
            -flto
            -march=native
            -ffast-math
            -funroll-loops
            -ftree-vectorize
    )

    target_compile_options(CmlContentBasedFiltering PRIVATE ${COMMON_COMPILE_OPTIONS})

    if(IS_NAIVE)
        add_definitions(-DNAIVE)
    endif ()

    add_executable(benchmark_matrix benchmarks/benchmark_matrix.cpp)

    target_compile_options(benchmark_matrix PRIVATE ${COMMON_COMPILE_OPTIONS})

    target_link_libraries(
            benchmark_matrix
            benchmark::benchmark
            CmlContentBasedFiltering
            TBB::tbb
    )

endif()


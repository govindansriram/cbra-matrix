#!/bin/bash
set -e

# Define the build directory
BUILD_DIR="build"
LIB_DIR="lib"
LOG_DIR="logs"
BENCHMARK=false
THREAD_SANITIZE=false

# Function to display usage
usage() {
    echo "Usage: $0 [-t THREAD_SANITIZE] [-b BENCHMARK]"
    echo "  -b      benchmark mode"
    echo "  -t      test with threads"
    exit 1
}

while getopts "bt" opt; do
    case $opt in
        b) BENCHMARK=true ;;
        t) THREAD_SANITIZE=true ;;
    esac
done

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating $BUILD_DIR directory..."
    mkdir "$BUILD_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating $LOG_DIR directory..."
    mkdir "$LOG_DIR"
fi

cd "$BUILD_DIR" || exit

if [ "$BENCHMARK" = true ]; then
    echo "Running benchmarks"
    cmake -DENABLE_TESTING=OFF ..
else
    if [ "$THREAD_SANITIZE" = true ]; then
      echo "Running thread sanitization"
      cmake -DENABLE_TESTING=ON -DIS_THREAD=ON ..
    else
      echo "Running address sanitization"
      cmake -DENABLE_TESTING=ON -DIS_THREAD=OFF ..
    fi
fi

make

echo
echo "finished build"
echo

if $BENCHMARK; then
    pattern="benchmark*"
else
    pattern="test*"
fi

for exe in $pattern; do
    # Check if it's an executable file
    if [ -x "$exe" ] && [ ! -d "$exe" ]; then
        echo "Running $exe..."
        echo "----------------------------------------------------"
        echo

        if $BENCHMARK; then
            ./"$exe" --benchmark_counters_tabular=true --benchmark_format=console --benchmark_out=../benchmarks/reports/"$exe".json
        else
            ./"$exe"
        fi

        echo
    fi
done
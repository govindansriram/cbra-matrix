#!/bin/bash
set -e

# Define the build directory
BUILD_DIR="build"
LIB_DIR="lib"
LOG_DIR="logs"


# Function to process a single valgrind log file
process_valgrind_log() {
    local file=$1
    echo
    echo "Analyzing: $file"
    echo "----------------------------------------"

    awk '/HEAP SUMMARY:/ { print; found=1 } found { print }' "$file"
}

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating $BUILD_DIR directory..."
    mkdir "$BUILD_DIR"
fi

## Check if the build directory exists
#if [ ! -d "$LIB_DIR" ]; then
#    echo "Creating $LIB_DIR directory..."
#    mkdir "$LIB_DIR"
#fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating $LOG_DIR directory..."
    mkdir "$LOG_DIR"
fi

cd "$BUILD_DIR" || exit

cmake ..

make

echo
echo "finished build"
echo

for test_executable in test*; do
    # Check if it's an executable file
    if [ -x "$test_executable" ] && [ ! -d "$test_executable" ]; then
        echo "Running $test_executable..."
        echo "----------------------------------------------------"
        echo

        valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
                 --verbose --log-file="../$LOG_DIR/$test_executable.valgrind.txt" ./"$test_executable"

        process_valgrind_log "../$LOG_DIR/$test_executable.valgrind.txt"

        echo
    fi
done
# Use an official Ubuntu base image
FROM ubuntu:22.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    curl \
    valgrind \
    libtbb-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Google Test
RUN git clone https://github.com/google/googletest.git /tmp/googletest && \
    cd /tmp/googletest && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_GMOCK=ON -S . -B build && \
    cmake --build build --target install && \
    rm -rf /tmp/googletest

# Install Google Benchmark
RUN git clone https://github.com/google/benchmark.git /tmp/benchmark && \
    git clone https://github.com/google/googletest.git /tmp/benchmark/googletest && \
    cd /tmp/benchmark && \
    cmake -DCMAKE_BUILD_TYPE=Release -S . -B build && \
    cmake --build build --target install && \
    rm -rf /tmp/benchmark

# Set C++ standard to C++17
ENV CXXFLAGS="-std=c++17"

# Default command
CMD ["/bin/bash"]

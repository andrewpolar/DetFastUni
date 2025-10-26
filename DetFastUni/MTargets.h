#pragma once
#include <vector>

class MTargets {
public:
    static inline std::vector<double> data;
    static inline size_t rows = 0;
    static inline size_t cols = 0;

    static void resize(size_t r, size_t c) {
        rows = r;
        cols = c;
        data.resize(r * c);
    }

    static double& at(size_t i, size_t j) {
        return data[i * cols + j];
    }
};

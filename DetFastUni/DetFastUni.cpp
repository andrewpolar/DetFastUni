//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

//VS compile as C++20

//This is demo of one concept of parallel Newton-Kaczmarz for Kolmogorov-Arnold networks. 
//Training to predict determinants of random 5 by 5 matrices on 10'000'000 records. Typical time near 5 minutes. 
//Code is portable to Linux.
//Makefile
//# Compiler and flags
//CXX = g++
//CXXFLAGS = -O2 -std=c++17 -Wall -pthread
//LDFLAGS = -pthread

//# Target name (final executable)
//TARGET = DetFastUni

//# Source files
//SRCS = DetFastUni.cpp

//# Object files
//OBJS = $(SRCS:.cpp=.o)

//# Default rule
//$(TARGET): $(OBJS)
//	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

//# Compile .cpp to .o
//%.o: %.cpp
//	$(CXX) $(CXXFLAGS) -c $< -o $@

//# Clean rule
//clean:
//	rm -f $(OBJS) $(TARGET)

#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include "KANAddend.h"
#include "Helper.h"
#include "MTargets.h"
#include "TwoPhaseBarrier.h"

static int MAX_LOOPS = 1200;

// Worker: persistent thread per group
void worker_thread(int groupId,
    const std::vector<KANAddend*>& group_addends,      
    const std::vector<KANAddend*>& group_frozen_ptrs, 
    const std::vector<std::vector<double>>& features,
    TwoPhaseBarrier& barrier,
    int nGroups, int Batch, double mu,
    int nTrainingRecords)
{
    const int nSub = (int)group_addends.size();
    int loop = 0;

    while (true) {
        //Phase A: compute using frozen snapshot (write MTargets) 
        int start = (loop * Batch) % nTrainingRecords;
        for (int i = 0; i < Batch; ++i) {
            int idx = (start + i) % nTrainingRecords;
            double model = 0.0;
            for (int j = 0; j < nSub; ++j) {
                model += group_frozen_ptrs[j]->ComputeUsingInput(features[idx]);
            }
            MTargets::at(groupId, i) = model;
        }

        //signal to main that this worker finished computing MTargets for this loop
        barrier.arriveUpper(loop);

        //Phase B: wait for main's serial reshape, then apply updates 
        for (int local_epoch = 0; local_epoch < 2; ++local_epoch) {
            for (int i = 0; i < Batch; ++i) {
                int idx = (start + i) % nTrainingRecords;
                double model = 0.0;
                for (int j = 0; j < nSub; ++j) {
                    model += group_addends[j]->ComputeUsingInput(features[idx]);
                }
                double residual = MTargets::at(groupId, i) - model;
                residual *= mu;
                for (int j = 0; j < nSub; ++j) {
                    group_addends[j]->UpdateUsingMemory(residual);
                }
            }
        }

        //signal to main that updates for this worker are done (lower phase)
        barrier.arriveLower(loop);

        //self termination
        ++loop;
        if (loop >= MAX_LOOPS) break;
    }
}

int main() {
    srand((unsigned int)time(NULL));

    // Dataset + hyperparameters
    // Matrix 5 * 5
    const int nTrainingRecords = 10'000'000;
    const int nValidationRecords = 2'000'000;
    const int nMatrixSize = 5;
    const double min = 0.0;
    const double max = 10.0;

    const int nGroups = 13;  //it is equal to number of threads
    const int nAddendsInGroup = 16;
    const int nAddends = nGroups * nAddendsInGroup;
    const double mu = 0.1 / nAddends;
    const int Batch = 16'000;
    const int nLoopsBeforeValidation = 100;
    const double termination = 0.87;
    const int nPointsInner = 5;
    const int nPointsOuter = 22;
    //////////////////////////////////////

    //// Dataset + hyperparameters
    //// Matrix 4 * 4
    //const int nTrainingRecords = 100'000;
    //const int nValidationRecords = 20'000;
    //const int nMatrixSize = 4;
    //const double min = 0.0;
    //const double max = 10.0;

    //const int nGroups = 6;  //it is equal to number of threads
    //const int nAddendsInGroup = 9;
    //const int nAddends = nGroups * nAddendsInGroup;
    //const double mu = 0.2 / nAddends;
    //const int Batch = 10'000;
    //const double termination = 0.97;
    //const int nPointsInner = 5;
    //const int nPointsOuter = 22;
    //const int nLoopsBeforeValidation = 20;

    printf("Data generation started ...\n");
    int nFeatures = nMatrixSize * nMatrixSize;
    auto features = Helper::GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto targets = Helper::ComputeDeterminantTarget(features, nMatrixSize, nTrainingRecords);

    auto features_validation = Helper::GenerateInput(nValidationRecords, nFeatures, min, max);
    auto targets_validation = Helper::ComputeDeterminantTarget(features_validation, nMatrixSize, nValidationRecords);
    printf("Dataset is generated\n");

    // Timing
    auto parallel_start = std::chrono::high_resolution_clock::now();

    // feature limits
    std::vector<double> xmin(nFeatures, min), xmax(nFeatures, max);
    double targetMin = *std::min_element(targets.begin(), targets.end());
    double targetMax = *std::max_element(targets.begin(), targets.end());

    // create addends (the single authoritative model that gets updated)
    std::vector<std::unique_ptr<KANAddend>> addends;
    addends.reserve(nAddends);
    for (int i = 0; i < nAddends; ++i) {
        addends.push_back(std::make_unique<KANAddend>(xmin, xmax, targetMin / nAddends, targetMax / nAddends, nPointsInner, nPointsOuter, nFeatures));
    }

    //first concurrent stage is pretraining, random pairs are trained individually, it is only good for very approximate model
    printf("Pretraining started ...\n");
    auto pairs = Helper::MakePairs(nAddends);

    //pairs of addends
    std::vector<std::thread> pre_threads;
    for (auto& p : pairs) {
        int first = p.first;
        int second = p.second;
        pre_threads.emplace_back([first, second, &features, &targets, mu, nTrainingRecords, &addends]() {
            for (int epoch = 0; epoch < 2; ++epoch) {
                for (int i = 0; i < nTrainingRecords; ++i) {
                    double model = addends[first]->ComputeUsingInput(features[i]);
                    model += addends[second]->ComputeUsingInput(features[i]);
                    double residual = targets[i] - model;
                    residual *= mu;
                    addends[first]->UpdateUsingMemory(residual);
                    addends[second]->UpdateUsingMemory(residual);
                }
            }
            });
    }
    for (auto& t : pre_threads) {
        t.join();
    }
    auto parallel_end_pre = std::chrono::high_resolution_clock::now();
    auto parallel_ms_pre = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_end_pre - parallel_start);
    printf("Pretraining ended, time %2.1f\n", static_cast<double>(parallel_ms_pre.count()));

    MTargets::resize(nGroups, Batch);

    //Pre-create per-group pointers to the addends (each worker uses its slice to update)
    std::vector<std::vector<KANAddend*>> group_addends(nGroups);
    for (int g = 0; g < nGroups; ++g) {
        int first = g * nAddendsInGroup;
        for (int j = 0; j < nAddendsInGroup; ++j)
            group_addends[g].push_back(addends[first + j].get());
    }

    //Create initial frozens_0 BEFORE launching workers (snapshot of addends at t=0)
    std::vector<std::unique_ptr<KANAddend>> frozens;
    frozens.reserve(nAddends);
    for (int i = 0; i < nAddends; ++i)
        frozens.push_back(std::make_unique<KANAddend>(*addends[i])); // initial snapshot

    //frozen pointers per group (workers will read from these; main will update them at end of each loop)
    std::vector<std::vector<KANAddend*>> frozen_group_ptrs(nGroups, std::vector<KANAddend*>(nAddendsInGroup, nullptr));
    // init frozen_group_ptrs to point into frozens (frozens_0)
    for (int g = 0; g < nGroups; ++g) {
        int first = g * nAddendsInGroup;
        for (int j = 0; j < nAddendsInGroup; ++j)
            frozen_group_ptrs[g][j] = frozens[first + j].get();
    }

    //Barrier and threads
    TwoPhaseBarrier barrier(nGroups);
    std::vector<std::thread> threads;
    threads.reserve(nGroups);

    // Launch persistent worker threads (each receives *pointers to frozen_group_ptrs[g]*)
    for (int g = 0; g < nGroups; ++g) {
        threads.emplace_back(worker_thread,
            g,
            std::cref(group_addends[g]),       //actual addends (workers update these)
            std::cref(frozen_group_ptrs[g]),   //pointer vector to frozens (will be updated by main each loop)
            std::cref(features),
            std::ref(barrier),
            nGroups, Batch, mu,
            nTrainingRecords);
    }

    // MAIN loop: follows the pattern
    // - waitAllUpper (workers have produced MTargets for this loop)
    // - serial reshape (adjust MTargets)
    // - releaseUpper (lets workers run updates)
    // - waitAllLower (workers finished updates)
    // - rescale comparing current addends with frozens_t
    // - create new frozens_{t+1} for next loop and repoint frozen_group_ptrs accordingly
    std::vector<double> predicted_target(nValidationRecords);
    for (int loop = 0; loop < MAX_LOOPS; ++loop) {
        // --- wait for workers to finish computing MTargets (they computed using frozens for this loop) ---
        barrier.waitAllUpper();

        // --- serial reshape: compute const_part and reassign MTargets (exactly as old logic) ---
        int start = (loop * Batch) % nTrainingRecords;
        for (int i = 0; i < Batch; ++i) {
            int idx = (start + i) % nTrainingRecords;
            double model = 0.0;
            for (int g = 0; g < nGroups; ++g)
                model += MTargets::at(g, i);
            double const_part = targets[idx] - model;
            for (int g = 0; g < nGroups; ++g) {
                MTargets::at(g, i) = const_part + MTargets::at(g, i);
            }
        }

        // --- allow workers to proceed to update addends using MTargets ---
        barrier.releaseUpper();

        // --- wait for all workers to finish updating their addends ---
        barrier.waitAllLower();

        // --- rescale: compare current addends to frozens (snapshot used for compute this loop) ---
        for (int i = 0; i < nAddends; ++i) {
            addends[i]->RescaleAddend(*frozens[i], nGroups);
        }

        // --- validation / logging (optional) ---
        if ((loop % nLoopsBeforeValidation == 0 && loop > 0) || loop == MAX_LOOPS - 1) {
            auto parallel_end = std::chrono::high_resolution_clock::now();
            auto parallel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_end - parallel_start);
            for (size_t i = 0; i < nValidationRecords; ++i) {
                predicted_target[i] = 0.0;
                for (size_t j = 0; j < addends.size(); ++j) {
                    predicted_target[i] += addends[j]->ComputeUsingInput(features_validation[i], true);
                }
            }
            double pearson = Helper::Pearson(predicted_target, targets_validation, (int)predicted_target.size());
            printf("Parallel Newton-Kaczmarz, loop %d, batch %d, validation Pearson %4.3f, training time ms %2.1f\n",
                loop, Batch, pearson, static_cast<double>(parallel_ms.count()));
            if (pearson >= termination) {
                //this terminates all threads nicely
                MAX_LOOPS = 0;
            }
        }

        // --- create frozens_{t+1} = snapshot(addends) for the next loop
        // Replace the old frozens vector with a fresh snapshot of current addends.
        // Important: keep the old frozens alive until after Rescale (we did), now we overwrite it.
        std::vector<std::unique_ptr<KANAddend>> next_frozens;
        next_frozens.reserve(nAddends);
        for (int i = 0; i < nAddends; ++i)
            next_frozens.push_back(std::make_unique<KANAddend>(*addends[i]));

        // swap frozens and update frozen_group_ptrs to point to new frozens entries
        frozens.swap(next_frozens);
        // next_frozens will be destroyed here (old frozens), which is fine (we already rescaled against old frozens)

        for (int g = 0; g < nGroups; ++g) {
            int first = g * nAddendsInGroup;
            for (int j = 0; j < nAddendsInGroup; ++j)
                frozen_group_ptrs[g][j] = frozens[first + j].get();
        }

        // --- release workers to start next iteration ---
        barrier.releaseLower();
    }

    // join worker threads (clean exit)
    for (auto& t : threads) if (t.joinable()) t.join();

    printf("Training complete.\n");
    return 0;
}


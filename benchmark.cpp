#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <benchmark/benchmark.h>

using namespace MicroTorch;

inline void BM_ResRNNForward(benchmark::State& state)
{
    constexpr int num_samples = 512;
    constexpr int sampling_freq = 44100;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/resrnn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_WaveNetForward(benchmark::State& state) {
    constexpr int num_samples = 512;
    constexpr int sampling_freq = 44100;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/wavenet.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

// Register the function as a benchmark
BENCHMARK(BM_ResRNNForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNetForward)->Unit(benchmark::kMillisecond);

// Run the benchmark
BENCHMARK_MAIN();
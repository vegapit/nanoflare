#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <benchmark/benchmark.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace MicroTorch;

inline void BM_ResRNNForward(benchmark::State& state)
{
    constexpr int num_samples = 512;
    //constexpr int sampling_freq = 44100;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/resrnn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_WaveNetForward(benchmark::State& state)
{
    constexpr int num_samples = 512;
    //constexpr int sampling_freq = 44100;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/wavenet.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_ResRNNLibtorchForward(benchmark::State& state)
{
    constexpr int num_samples = 512;
    //constexpr int sampling_freq = 44100;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/resrnn.torchscript");
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));
    inputs.push_back(hc);

    torch::NoGradGuard no_grad;
    module.eval();
    
    for (auto _ : state)
        module.forward(inputs);
}

// Register the function as a benchmark
BENCHMARK(BM_ResRNNForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNetForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResRNNLibtorchForward)->Unit(benchmark::kMillisecond);

// Run the benchmark
BENCHMARK_MAIN();
#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <benchmark/benchmark.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace MicroTorch;

inline void BM_ResLSTMForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/reslstm.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_ResLSTMLibtorchForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/reslstm.torchscript");
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

inline void BM_ResGRUForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/resgru.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_ResGRULibtorchForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/resgru.torchscript");
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));
    inputs.push_back(torch::zeros({1, 1, 64}));

    torch::NoGradGuard no_grad;
    module.eval();
    
    for (auto _ : state)
        module.forward(inputs);
}

inline void BM_TCNForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/tcn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);

}

inline void BM_TCNLibtorchForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/tcn.torchscript");
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    torch::NoGradGuard no_grad;
    module.eval();
    
    for (auto _ : state)
        module.forward(inputs);
}

inline void BM_WaveNetForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/wavenet.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
        obj->forward(x);
}

inline void BM_WaveNetLibtorchForward(benchmark::State& state)
{
    constexpr int num_samples = 512;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/wavenet.torchscript");
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    torch::NoGradGuard no_grad;
    module.eval();
    
    for (auto _ : state)
        module.forward(inputs);
}

// Register the function as a benchmark

BENCHMARK(BM_ResGRUForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResGRULibtorchForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResLSTMForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResLSTMLibtorchForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TCNForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TCNLibtorchForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNetForward)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNetLibtorchForward)->Unit(benchmark::kMillisecond);

// Run the benchmark
BENCHMARK_MAIN();
#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <benchmark/benchmark.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace MicroTorch;

inline void BM_ResLSTM(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/reslstm.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        obj->forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_ResLSTMLibtorch(benchmark::State& state)
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
    {
        auto start = std::chrono::high_resolution_clock::now();
        module.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_ResGRU(benchmark::State& state)
{
    constexpr int num_samples = 512;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/resgru.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        obj->forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_ResGRULibtorch(benchmark::State& state)
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
    {
        auto start = std::chrono::high_resolution_clock::now();
        module.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_MicroTCN(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/microtcn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        obj->forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }

}

inline void BM_MicroTCNLibtorch(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load("../test_data/microtcn.torchscript");
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
    {
        auto start = std::chrono::high_resolution_clock::now();
        module.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_TCN(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/tcn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        obj->forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }

}

inline void BM_TCNLibtorch(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

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
    {
        auto start = std::chrono::high_resolution_clock::now();
        module.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_WaveNet(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream("../test_data/wavenet.json");
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        obj->forward(x);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

inline void BM_WaveNetLibtorch(benchmark::State& state)
{
    constexpr int num_samples = 512 + 256;

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
    {
        auto start = std::chrono::high_resolution_clock::now();
        module.forward(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
        state.counters["RTF[512@44.1kHz]"] = double(num_samples) / ( 44100.0 * elapsed_seconds.count() );
    }
}

// Register the function as a benchmark

BENCHMARK(BM_ResGRU)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResGRULibtorch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResLSTM)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ResLSTMLibtorch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MicroTCN)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MicroTCNLibtorch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TCN)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TCNLibtorch)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNet)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_WaveNetLibtorch)->UseManualTime()->Unit(benchmark::kMillisecond);

// Run the benchmark
BENCHMARK_MAIN();
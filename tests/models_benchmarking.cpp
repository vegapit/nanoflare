#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "nanoflare/ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include "filesystem.h"

using namespace NanoFlare;

constexpr int num_samples = 512;

TEST_CASE("ConvWaveShaper") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/convwaveshaper.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    BENCHMARK("ConvWaveShaper") {
        return obj->forward(x);
    };
}

TEST_CASE("ConvWaveShaper TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/convwaveshaper.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);
    
    BENCHMARK("ConvWaveShaper TorchScript") {
        return module.forward(inputs);
    };
}

TEST_CASE("MicroTCN") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/microtcn.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);
    
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);
    BENCHMARK("MicroTCN") {
        return obj->forward(x);
    };
}

TEST_CASE("MicroTCN TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/microtcn.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);

    BENCHMARK("MicroTCN TorchScript") {
        return module.forward(inputs);
    };
}

TEST_CASE("ResGRU") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/resgru.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);
    BENCHMARK("ResGRU") {
        return obj->forward(x);
    };
}

TEST_CASE("ResGRU TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/resgru.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));
    inputs.push_back( torch::zeros({1, 1, 64}) );

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);

    BENCHMARK("ResGRU TorchScript") {
        return module.forward(inputs);
    };
}

TEST_CASE("ResLSTM") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/reslstm.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);
    BENCHMARK("ResLSTM") {
        return obj->forward(x);
    };
}

TEST_CASE("ResLSTM TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/reslstm.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };
    inputs.push_back( hc );

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);

    BENCHMARK("ResLSTM TorchScript") {
        return module.forward(inputs);
    };
}

TEST_CASE("TCN") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/tcn.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);
    
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);
    BENCHMARK("TCN") {
        return obj->forward(x);
    };
}

TEST_CASE("TCN TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/tcn.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);

    BENCHMARK("TCN TorchScript") {
        return module.forward(inputs);
    };
}

TEST_CASE("WaveNet") {
    std::shared_ptr<BaseModel> obj;
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/wavenet.json");

    std::ifstream fstream( modelPath.c_str() );
    nlohmann::json data = nlohmann::json::parse(fstream);
    ModelBuilder::fromJson(data, obj);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);
    BENCHMARK("WaveNet") {
        return obj->forward(x);
    };
}

TEST_CASE("WaveNet TorchScript") {
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path( "tests/data/wavenet.torchscript" );

    torch::set_num_threads(1);
    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    module.eval();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    // Warm-up
    for(auto i = 0; i < 10; ++i)
        module.forward(inputs);

    BENCHMARK("WaveNet TorchScript") {
        return module.forward(inputs);
    };
}
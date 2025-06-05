#include <catch2/catch_test_macros.hpp>
#include "nanoflare/ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include "filesystem.h"

using namespace NanoFlare;

inline RowMatrixXf torch_to_eigen(const torch::Tensor& t)
{
    auto acc = t.accessor<float, 2>();
    RowMatrixXf res( acc.size(0), acc.size(1) );
    for(auto i = 0; i < acc.size(0); i++)
        for(auto j = 0; j < acc.size(1); j++)
            res(i,j) = acc[i][j];
    return res;
}

constexpr int num_samples = 2048;

TEST_CASE("ConvWaveshaper Test", "[ConvWaveshaper]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/convwaveshaper.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/convwaveshaper.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() < 1e-4 );
}

TEST_CASE("MicroTCN Test", "[MicroTCN]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/microtcn.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/microtcn.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() < 1e-4 );
}

TEST_CASE("ResGRU Test", "[ResGRU]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/resgru.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/resgru.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( torch::zeros({1, 1, 64}) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() < 1e-4 );
}

TEST_CASE("ResLSTM Test", "[ResLSTM]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/reslstm.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/reslstm.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( hc );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() < 1e-4 );
}

TEST_CASE("TCN Test", "[TCN]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/tcn.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/tcn.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() < 1e-4 );
}

TEST_CASE("WaveNet Test", "[WaveNet]")
{
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/wavenet.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/wavenet.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );
    
    REQUIRE( (pred - target).norm() < 1e-4 );
}
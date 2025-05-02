#include <catch2/catch_test_macros.hpp>
#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace MicroTorch;

inline RowMatrixXf torch_to_eigen(const torch::Tensor& t) {
    auto acc = t.accessor<float, 2>();
    RowMatrixXf res( acc.size(0), acc.size(1) );
    for(auto i = 0; i < acc.size(0); i++)
        for(auto j = 0; j < acc.size(1); j++)
            res(i,j) = acc[i][j];
    return res;
}

bool microtcn_match()
{
    constexpr int num_samples = 4096;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/microtcn.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/microtcn.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-4 );
}

bool resgru_match()
{
    constexpr int num_samples = 4096;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/resgru.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/resgru.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( torch::zeros({1, 1, 64}) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-4 );
}

bool reslstm_match()
{
    constexpr int num_samples = 2048;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/reslstm.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/reslstm.torchscript");
    
    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( hc );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-4 );
}

bool convwaveshaper_match()
{
    constexpr int num_samples = 2048;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/convwaveshaper.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/convwaveshaper.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-4 );
}

bool tcn_match()
{
    constexpr int num_samples = 4096;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/tcn.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/tcn.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-4 );
}

bool wavenet_match()
{
    constexpr int num_samples = 4096;

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../data/wavenet.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj->forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../data/wavenet.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );
    
    return ( (pred - target).norm() < 1e-4 );
}

TEST_CASE("MicroTCN Test", "[microtcn_match]")
{
    REQUIRE( microtcn_match() );
}

TEST_CASE("ResGRU Test", "[resgru_match]")
{
    REQUIRE( resgru_match() );
}

TEST_CASE("ResLSTM Test", "[reslstm_match]")
{
    REQUIRE( reslstm_match() );
}

TEST_CASE("ConvWaveshaper Test", "[convwaveshaper_match]")
{
    REQUIRE( convwaveshaper_match() );
}

TEST_CASE("TCN Test", "[tcn_match]")
{
    REQUIRE( tcn_match() );
}

TEST_CASE("WaveNet Test", "[wavenet_match]")
{
    REQUIRE( wavenet_match() );
}
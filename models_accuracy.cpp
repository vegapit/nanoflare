#include <catch2/catch_test_macros.hpp>
#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>

using namespace MicroTorch;

bool resgru_match()
{
    constexpr int num_samples = 44100;

    auto torch_data = torch::randn({1, 1, num_samples});
    Eigen::Map<RowMatrixXf> eigen_data( torch_data.data_ptr<float>(), 1, num_samples );

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../test_data/resgru.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    torch::jit::script::Module module = torch::jit::load("../test_data/resgru.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);
    inputs.push_back(torch::zeros({1, 1, 64}));

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    Eigen::Map<RowMatrixXf> target( torch_res.data_ptr<float>(), 1, num_samples );

    auto pred = obj->forward( eigen_data );

    return ( (pred - target).norm() < 1e-3 );
}

bool tcn_match()
{
    constexpr int num_samples = 44100;

    auto torch_data = torch::randn({1, 1, num_samples});
    Eigen::Map<RowMatrixXf> eigen_data( torch_data.data_ptr<float>(), 1, num_samples );

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../test_data/tcn.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    torch::jit::script::Module module = torch::jit::load("../test_data/tcn.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    Eigen::Map<RowMatrixXf> target( torch_res.data_ptr<float>(), 1, num_samples );

    auto pred = obj->forward( eigen_data );

    std::cout << (pred - target).norm() << std::endl;

    return ( (pred - target).norm() < 1e-3 );
}

bool wavenet_match()
{
    constexpr int num_samples = 44100;

    auto torch_data = torch::randn({1, 1, num_samples});
    Eigen::Map<RowMatrixXf> eigen_data( torch_data.data_ptr<float>(), 1, num_samples );

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file("../test_data/wavenet.json");
    ModelBuilder::fromJson( nlohmann::json::parse(model_file), obj );

    torch::jit::script::Module module = torch::jit::load("../test_data/wavenet.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    Eigen::Map<RowMatrixXf> target( torch_res.data_ptr<float>(), 1, num_samples );

    auto pred = obj->forward( eigen_data );
    
    return ( (pred - target).norm() < 1e-3 );
}

TEST_CASE("ResGRU Test", "[resgru_match]")
{
    REQUIRE( resgru_match() );
}

TEST_CASE("TCN Test", "[tcn_match]")
{
    REQUIRE( tcn_match() );
}

TEST_CASE("WaveNet Test", "[wavenet_match]")
{
    REQUIRE( wavenet_match() );
}
#include <catch2/catch_all.hpp>
#include "nanoflare/ModelBuilder.h"
#include "nanoflare/BuiltinModels.h"
#include "nanoflare/models/BaseModel.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <filesystem>
#include <iostream>

using namespace Nanoflare;
using Catch::Approx;

inline RowMatrixXf torch_to_eigen_matrix(const torch::Tensor& t)
{
    auto acc = t.accessor<float, 2>();
    RowMatrixXf res( acc.size(0), acc.size(1) );
    for(auto i = 0; i < acc.size(0); i++)
        for(auto j = 0; j < acc.size(1); j++)
            res(i,j) = acc[i][j];
    return res;
}

inline Eigen::RowVectorXf torch_to_eigen_vector(const torch::Tensor& t)
{
    auto acc = t.accessor<float, 1>();
    Eigen::RowVectorXf res( acc.size(0) );
    for(auto i = 0; i < acc.size(0); i++)
        res(i) = acc[i];
    return res;
}

constexpr int num_samples = 2048;

TEST_CASE("HammersteinWiener Test", "[HammersteinWiener]")
{   
    std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/hammersteinwiener.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/hammersteinwiener.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}

TEST_CASE("MicroTCN Test", "[MicroTCN]")
{
        std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/microtcn.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/microtcn.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}

TEST_CASE("ResGRU Test", "[ResGRU]")
{
    std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/resgru.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/resgru.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( torch::zeros({1, 1, 64}) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto outputs = module.forward( inputs ).toTuple();
    auto torch_res = outputs->elements()[0].toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}

TEST_CASE("ResLSTM Test", "[ResLSTM]")
{
    std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/reslstm.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/reslstm.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );
    inputs.push_back( hc );

    torch::NoGradGuard no_grad;
    module.eval();

    auto outputs = module.forward( inputs ).toTuple();
    auto torch_res = outputs->elements()[0].toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}

TEST_CASE("TCN Test", "[TCN]")
{
    std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/tcn.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/tcn.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );

    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}

TEST_CASE("WaveNet Test", "[WaveNet]")
{
    std::filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= std::filesystem::path("tests/data/wavenet.json");
    std::filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= std::filesystem::path("tests/data/wavenet.torchscript");

    std::shared_ptr<BaseModel> obj;
    std::ifstream model_file( modelPath.c_str() );
    ModelBuilder::getInstance().buildModel(nlohmann::json::parse(model_file), obj );

    auto torch_data = torch::randn({1, num_samples});
    auto eigen_data = torch_to_eigen_matrix( torch_data );
    RowMatrixXf pred = RowMatrixXf::Zero(1, num_samples);
    obj->forward( eigen_data, pred );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen_matrix( torch_res.squeeze(0) );
    
    REQUIRE( (pred - target).norm() == Approx(0.0).margin(1e-4) );
}
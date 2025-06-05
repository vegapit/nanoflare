#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include "filesystem.h"

#include "nanoflare/layers/ConvClipper.h"
#include "nanoflare/layers/CausalDilatedConv1d.h"
#include "nanoflare/layers/GRU.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/MicroTCNBlock.h"
#include "nanoflare/layers/PlainSequential.h"
#include "nanoflare/layers/ResidualBlock.h"
#include "nanoflare/layers/TCNBlock.h"

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

TEST_CASE("CausalDilatedConv1d Test", "[CausalDilatedConv1d]")
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t dilation = 2;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/causaldilatedconv1d.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/causaldilatedconv1d.torchscript");

    CausalDilatedConv1d obj(inChannels, outChannels, kernelSize, true, dilation);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res );

    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("ConvClipper Test", "[ConvClipper]")
{
    size_t kernelSize = 12;
    size_t dilation = 4;
    size_t seqLength = 64;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/convclipper.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/convclipper.torchscript");

    ConvClipper obj(1, 1, kernelSize, dilation);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ 1, long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res );

    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("MicroTCNBlock Test", "[MicroTCNBlock]")
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t dilation = 2;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/microtcnblock.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/microtcnblock.torchscript");

    MicroTCNBlock obj(inChannels, outChannels, kernelSize, dilation);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );
    
    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("PlainSequential Test", "[PlainSequential]")
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t hiddenChannels = 8;
    size_t numHiddenLayers = 3;
    size_t batchSize = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/plainsequential.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/plainsequential.torchscript");

    PlainSequential obj(inChannels, outChannels, hiddenChannels, numHiddenLayers);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(batchSize), long(inChannels)});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res );

    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("ResidualBlock Test", "[ResidualBlock]")
{
    size_t numChannels = 7;
    size_t kernelSize = 3;
    size_t dilation = 2;
    bool gated = true;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/residualblock.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/residualblock.torchscript");

    ResidualBlock obj(numChannels, kernelSize, dilation, gated);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(numChannels), long(seqLength)});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data ).second;

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTuple()->elements();
    auto target = torch_to_eigen( torch_res[1].toTensor().squeeze(0) );
    
    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("TCNBlock Test", "[TCNBlock]")
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t dilation = 2;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/tcnblock.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/tcnblock.torchscript");

    TCNBlock obj(inChannels, outChannels, kernelSize, dilation);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );
    
    REQUIRE( (pred - target).norm() < 1e-5 );
}

TEST_CASE("convolve1d Test", "[convolve1d]")
{
    Eigen::RowVectorXf x(5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    Eigen::RowVectorXf w(3);
    w << 1.f, 0.5f, -1.f;

    auto pred = convolve1d(x, w);

    Eigen::RowVectorXf target(3);
    target << -1.5f, -1.f, -0.5f;

    REQUIRE((pred - target).norm() < 1e-5);
}

TEST_CASE("dilatedcausalconvolve1d Test", "[dilatedcausalconvolve1d]")
{
    Eigen::RowVectorXf x(5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    Eigen::RowVectorXf w(3);
    w << 1.f, 0.5f, -1.f;

    auto pred = dilatedcausalconvolve1d(x, w, 2);

    Eigen::RowVectorXf target(5);
    target << 0.f, -1.f, -2.f, -2.5f, -3.0f;

    REQUIRE((pred - target).norm() < 1e-5);
}

TEST_CASE("dilate Test", "[dilate]")
{
    Eigen::RowVectorXf x(3);
    x << 1.f, 2.f, 3.f;

    auto pred = dilate(x, 2);

    Eigen::RowVectorXf target(5);
    target << 1.f, 0.f, 2.f, 0.f, 3.f;

    REQUIRE((pred - target).norm() < 1e-5);
}
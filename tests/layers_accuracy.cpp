#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include "filesystem.h"

#include "BatchNorm1d.h"
#include "Conv1d.h"
#include "ConvClipper.h"
#include "CausalDilatedConv1d.h"
#include "GRU.h"
#include "Linear.h"
#include "LSTM.h"
#include "MicroTCNBlock.h"
#include "PlainSequential.h"
#include "ResidualBlock.h"
#include "TCNBlock.h"

using namespace MicroTorch;

inline RowMatrixXf torch_to_eigen(const torch::Tensor& t) {
    auto acc = t.accessor<float, 2>();
    RowMatrixXf res( acc.size(0), acc.size(1) );
    for(auto i = 0; i < acc.size(0); i++)
        for(auto j = 0; j < acc.size(1); j++)
            res(i,j) = acc[i][j];
    return res;
}

bool batchnorm1d_pytorch_match()
{
    size_t numChannels = 11;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/batchnorm1d.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/batchnorm1d.torchscript");

    BatchNorm1d obj(numChannels);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({long(numChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    obj.apply( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );
    
    return ( (eigen_data - target).norm() < 1e-5 );
}

bool causaldilatedconv1d_pytorch_match()
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

    return ( (pred - target).norm() < 1e-5 );
}

bool conv1d_pytorch_match()
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/conv1d.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/conv1d.torchscript");

    Conv1d obj(inChannels, outChannels, kernelSize, true);
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

    return ( (pred - target).norm() < 1e-5 );
}

bool convclipper_pytorch_match()
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

    return ( (pred - target).norm() < 1e-5 );
}

bool gru_pytorch_match()
{
    size_t inputSize = 7;
    size_t hiddenSize = 11;
    size_t seqLength = 5;
    
    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/gru.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/gru.torchscript");

    GRU obj(inputSize, hiddenSize, true);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(seqLength), long(inputSize) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data );
    inputs.push_back( torch::zeros({ 1, long(hiddenSize)}) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTuple()->elements();
    auto target = torch_to_eigen( torch_res[0].toTensor() );

    return ( (pred - target).norm() < 1e-5 );
}

bool linear_pytorch_match()
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t batchSize = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/linear.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/linear.torchscript");

    Linear obj(inChannels, outChannels, true);
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
    
    return ( (pred - target).norm() < 1e-5 );
}

bool lstm_pytorch_match()
{
    size_t inputSize = 7;
    size_t hiddenSize = 11;
    size_t seqLength = 5;

    filesystem::path modelPath( PROJECT_SOURCE_DIR );
    modelPath /= filesystem::path("tests/data/lstm.json");
    filesystem::path tsPath( PROJECT_SOURCE_DIR );
    tsPath /= filesystem::path("tests/data/lstm.torchscript");

    LSTM obj(inputSize, hiddenSize, true);
    std::ifstream model_file( modelPath.c_str() );
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(seqLength), long(inputSize) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load( tsPath.c_str() );
    
    std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({ 1, long(hiddenSize)}) , torch::zeros({ 1, long(hiddenSize)}) };

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);
    inputs.push_back(hc);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTuple()->elements();
    auto target = torch_to_eigen( torch_res[0].toTensor() );

    return ( (pred - target).norm() < 1e-5 );
}

bool microtcnblock_pytorch_match()
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
    
    return ( (pred - target).norm() < 1e-5 );
}

bool plainsequential_pytorch_match()
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

    return ( (pred - target).norm() < 1e-5 );
}

bool residualblock_pytorch_match()
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
    
    return ( (pred - target).norm() < 1e-5 );
}

bool tcnblock_pytorch_match()
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
    
    return ( (pred - target).norm() < 1e-5 );
}

bool convolve1d_calculate()
{
    Eigen::RowVectorXf x(5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    Eigen::RowVectorXf w(3);
    w << 1.f, 0.5f, -1.f;

    auto pred = convolve1d(x, w);

    Eigen::RowVectorXf target(3);
    target << -1.5f, -1.f, -0.5f;

    return ((pred - target).norm() < 1e-5);
}

bool dilatedcausalconvolve1d_calculate()
{
    Eigen::RowVectorXf x(5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    Eigen::RowVectorXf w(3);
    w << 1.f, 0.5f, -1.f;

    auto pred = dilatedcausalconvolve1d(x, w, 2);

    Eigen::RowVectorXf target(5);
    target << 0.f, -1.f, -2.f, -2.5f, -3.0f;

    return ((pred - target).norm() < 1e-5);
}

bool dilate_calculate()
{
    Eigen::RowVectorXf x(3);
    x << 1.f, 2.f, 3.f;

    auto pred = dilate(x, 2);

    Eigen::RowVectorXf target(5);
    target << 1.f, 0.f, 2.f, 0.f, 3.f;

    return ((pred - target).norm() < 1e-5);
}

TEST_CASE("convolve1d Test", "[convolve1d_calculate]")
{
    REQUIRE( convolve1d_calculate() );
}

TEST_CASE("dilatedcausalconvolve1d Test", "[pad_calculate]")
{
    REQUIRE( dilatedcausalconvolve1d_calculate() );
}

TEST_CASE("dilate Test", "[dilate_calculate]")
{
    REQUIRE( dilate_calculate() );
}

TEST_CASE("BatchNorm1d Test", "[batchnorm1d_pytorch_match]")
{
    REQUIRE( batchnorm1d_pytorch_match() );
}

TEST_CASE("CausalDilatedConv1D Test", "[causaldilatedconv1d_pytorch_match]")
{
    REQUIRE( causaldilatedconv1d_pytorch_match() );
}

TEST_CASE("Conv1D Test", "[conv1d_pytorch_match]")
{
    REQUIRE( conv1d_pytorch_match() );
}

TEST_CASE("ConvClipper Test", "[convclipper_pytorch_match]")
{
    REQUIRE( convclipper_pytorch_match() );
}

TEST_CASE("GRU Test", "[gru_pytorch_match]")
{
    REQUIRE( gru_pytorch_match() );
}

TEST_CASE("Linear Test", "[linear_pytorch_match]")
{
    REQUIRE( linear_pytorch_match() );
}

TEST_CASE("LSTM Test", "[lstm_pytorch_match]")
{
    REQUIRE( lstm_pytorch_match() );
}

TEST_CASE("MicroTCNBlock Test", "[microtcnblock_pytorch_match]")
{
    REQUIRE( microtcnblock_pytorch_match() );
}

TEST_CASE("PlainSequential Test", "[plainsequential_pytorch_match]")
{
    REQUIRE( plainsequential_pytorch_match() );
}

TEST_CASE("ResidualBlock Test", "[residualblock_pytorch_match]")
{
    REQUIRE( residualblock_pytorch_match() );
}

TEST_CASE("TCNBlock Test", "[tcnblock_pytorch_match]")
{
    REQUIRE( tcnblock_pytorch_match() );
}
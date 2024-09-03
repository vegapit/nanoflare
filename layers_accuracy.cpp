#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>

#include "BatchNorm1d.h"
#include "Conv1d.h"
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

    BatchNorm1d obj(numChannels);
    std::ifstream model_file("../test_data/batchnorm1d.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({long(numChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/batchnorm1d.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back( torch_data.unsqueeze(0) );

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTensor();
    auto target = torch_to_eigen( torch_res.squeeze(0) );

    return ( (pred - target).norm() < 1e-5 );
}

bool causaldilatedconv1d_pytorch_match()
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t dilation = 2;
    size_t seqLength = 5;

    CausalDilatedConv1d obj(inChannels, outChannels, kernelSize, true, dilation);
    std::ifstream model_file("../test_data/causaldilatedconv1d.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/causaldilatedconv1d.torchscript");
    
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

    Conv1d obj(inChannels, outChannels, kernelSize, true);
    std::ifstream model_file("../test_data/conv1d.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/conv1d.torchscript");
    
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
    
    GRU obj(inputSize, hiddenSize, true);
    std::ifstream model_file("../test_data/gru.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(seqLength), long(inputSize) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/gru.torchscript");

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

    Linear obj(inChannels, outChannels, true);
    std::ifstream model_file("../test_data/linear.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(batchSize), long(inChannels)});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/linear.torchscript");
    
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
    
    LSTM obj(inputSize, hiddenSize, true);
    std::ifstream model_file("../test_data/lstm.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(seqLength), long(inputSize) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/lstm.torchscript");
    
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

    MicroTCNBlock obj(inChannels, outChannels, kernelSize, dilation);
    std::ifstream model_file("../test_data/microtcnblock.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/microtcnblock.torchscript");
    
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

    PlainSequential obj(inChannels, outChannels, hiddenChannels, numHiddenLayers);
    std::ifstream model_file("../test_data/plainsequential.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(batchSize), long(inChannels)});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/plainsequential.torchscript");
    
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

    ResidualBlock obj(numChannels, kernelSize, dilation, true, true, gated);
    std::ifstream model_file("../test_data/residualblock.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(numChannels), long(seqLength)});
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data ).second;

    torch::jit::script::Module module = torch::jit::load("../test_data/residualblock.torchscript");
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_data);

    torch::NoGradGuard no_grad;
    module.eval();

    auto torch_res = module.forward( inputs ).toTuple()->elements();
    auto target = torch_to_eigen( torch_res[1].toTensor() );
    
    return ( (pred - target).norm() < 1e-5 );
}

bool tcnblock_pytorch_match()
{
    size_t inChannels = 7;
    size_t outChannels = 11;
    size_t kernelSize = 3;
    size_t dilation = 2;
    size_t seqLength = 5;

    TCNBlock obj(inChannels, outChannels, kernelSize, dilation);
    std::ifstream model_file("../test_data/tcnblock.json");
    obj.loadStateDict( nlohmann::json::parse(model_file) );

    auto torch_data = torch::randn({ long(inChannels), long(seqLength) });
    auto eigen_data = torch_to_eigen( torch_data );
    auto pred = obj.forward( eigen_data );

    torch::jit::script::Module module = torch::jit::load("../test_data/tcnblock.torchscript");
    
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

bool pad_calculate()
{
    Eigen::RowVectorXf x(5);
    x << 1.f, 2.f, 3.f, 4.f, 5.f;

    auto pred = padLeft(x, 2);

    Eigen::RowVectorXf target(7);
    target << 0.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f;

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

TEST_CASE("pad Test", "[pad_calculate]")
{
    REQUIRE( pad_calculate() );
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
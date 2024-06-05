#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

#include "LSTM.h"
#include "GRU.h"
#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "ResidualBlock.h"
#include "Linear.h"

using namespace MicroTorch;

bool lstm_pytorch_match()
{
    size_t hiddenSize = 7;
    size_t inputSize = 1;

    std::ifstream f("../test_data/lstm.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    LSTM obj( inputSize , hiddenSize, true );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(2, 1);
    x << 0.5f, -0.5f;

    auto pred = obj.forward( x ); // X = [SeqLength, InputSize]
    
    std::cout << "LSTM Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2, 7);
    target <<  0.0318f,  0.0294f, -0.0346f,  0.0196f,  0.0069f,  0.0891f,  0.0012f, 
        0.0446f,  0.0446f, -0.0710f,  0.0026f, -0.0512f,  0.0483f, -0.0011f;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
}

bool gru_pytorch_match()
{
    size_t hiddenSize = 7;
    size_t inputSize = 1;

    std::ifstream f("../test_data/gru.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    GRU obj( inputSize , hiddenSize, true );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(2, 1);
    x << 0.5f, -0.5f;

    auto pred = obj.forward( x ); // X = [SeqLength, InputSize]
    
    std::cout << "GRU Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2, 7);
    target << -0.0626f, 0.1718f, 0.0554f, -0.1304f, 0.0460f, 0.0256f, 0.0403f, 
        -0.1066f, 0.1771f, 0.2300f, -0.1817f, 0.1646f, 0.1487, 0.1151;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
}

bool linear_pytorch_match()
{
    size_t inChannels = 3;
    size_t outChannels = 2;

    std::ifstream f("../test_data/linear.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    Linear obj( inChannels, outChannels, true);
    obj.loadStateDict( state_dict );

    RowMatrixXf x(2, 3);
    x << 1.f, -1.f, -2.f, -1.f, 0.f, 1.f;

    auto pred = obj.forward(x);

    std::cout << "Linear Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2, 2);
    target <<  1.6924642f,  -0.50574875f, -1.1378999f, 0.10293144f;

    return ( (pred - target).lpNorm<1>() < 1e-5 );
}

bool conv1d_pytorch_match()
{
    size_t inChannels = 1;
    size_t outChannels = 2;
    size_t kernelSize = 3;

    std::ifstream f("../test_data/conv1d.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    Conv1d obj( inChannels, outChannels, kernelSize, true);
    obj.loadStateDict( state_dict );

    RowMatrixXf x(1, 5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    auto pred = obj.forward(x);

    std::cout << "Conv1d Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2,3);
    target << -0.3752188f, -0.48619097f, -0.59716314f, 1.4530494f, 2.562778f, 3.6725063f;

    return ( (pred - target).lpNorm<1>() < 1e-5 );
}

bool causaldilatedconv1d_pytorch_match()
{
    size_t inChannels = 1;
    size_t outChannels = 2;
    size_t kernelSize = 3;
    bool bias = true;
    size_t dilation = 2;

    std::ifstream f("../test_data/causaldilatedconv1d.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    CausalDilatedConv1d obj( inChannels, outChannels, kernelSize, bias, dilation );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(1, 5);
    x << 0.f, 1.f, 2.f, 3.f, 4.f;

    auto pred = obj.forward(x);

    std::cout << "CausalDilatedConv1d Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2, 5);
    target << -0.2568827f, -0.2824603f, -0.30803785f, -0.7492607f, -1.1904836f,
        -0.26991627f, -0.07575521f, 0.11840585f, 0.30745503f, 0.49650422f;

    return ( (pred - target).lpNorm<1>() < 1e-5 );
}

bool residualblock_pytorch_match()
{
    size_t numChannels = 5;
    size_t kernelSize = 3;
    size_t dilation = 2;

    std::ifstream f("../test_data/residualblock.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    ResidualBlock obj( numChannels, kernelSize, dilation, true, true, true, Activation::TANH );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(5,1);
    x << 0.f, 0.1f, 0.2f, 0.3f, 0.4f;

    auto pred = std::get<0>( obj.forward(x) );

    std::cout << "ResidualBlock Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(5,1);
    target << -0.23491728f, 0.1363181f, 0.6830531f, 0.70318216f, 0.25027066f;

    return ( (pred - target).lpNorm<1>() < 1e-5 );
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

    float diff = (pred - target).lpNorm<1>();

    return (diff < 1e-5);
}

bool pad_calculate()
{
    Eigen::RowVectorXf x(5);
    x << 3.f, 2.f, 1.f, 2.f, 3.f;

    auto pred = pad(x, 2);

    Eigen::RowVectorXf target(9);
    target << 0.f, 0.f, 3.f, 2.f, 1.f, 2.f, 3.f, 0.f, 0.f;

    float diff = (pred - target).lpNorm<1>();

    return (diff < 1e-5);
}

bool dilate_calculate()
{
    Eigen::RowVectorXf x(3);
    x << 1.f, 2.f, 3.f;

    auto pred = dilate(x, 2);

    Eigen::RowVectorXf target(5);
    target << 1.f, 0.f, 2.f, 0.f, 3.f;

    float diff = (pred - target).lpNorm<1>();

    return (diff < 1e-5);
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

TEST_CASE("LSTM Test", "[lstm_pytorch_match]")
{
    REQUIRE( lstm_pytorch_match() );
}

TEST_CASE("GRU Test", "[gru_pytorch_match]")
{
    REQUIRE( gru_pytorch_match() );
}

TEST_CASE("Conv1D Test", "[conv1d_pytorch_match]")
{
    REQUIRE( conv1d_pytorch_match() );
}

TEST_CASE("CausalDilatedConv1D Test", "[causaldilatedconv1d_pytorch_match]")
{
    REQUIRE( causaldilatedconv1d_pytorch_match() );
}

TEST_CASE("ResidualBlock Test", "[residualblock_pytorch_match]")
{
    REQUIRE( residualblock_pytorch_match() );
}

TEST_CASE("Linear Test", "[linear_pytorch_match]")
{
    REQUIRE( linear_pytorch_match() );
}
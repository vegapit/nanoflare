#include <catch2/catch_test_macros.hpp>
#include "LSTM.h"
#include "GRU.h"
#include "Conv1d.h"
#include "CausalDilatedConv1d.h"
#include "ResidualBlock.h"
#include "Linear.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using namespace MicroTorch;

bool lstm_pytorch_match()
{
    int hiddenSize = 7;
    int inputSize = 1;

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
    int hiddenSize = 7;
    int inputSize = 1;

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
    int inChannels = 3;
    int outChannels = 2;

    std::ifstream f("../test_data/linear.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    Linear obj( inChannels, outChannels, true);
    obj.loadStateDict( state_dict );

    RowMatrixXf x(1, 3);
    x << 1.f, 0.5f, -1.f;

    auto pred = obj.forward(x);

    std::cout << "Linear Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(1, 2);
    target << -0.3685f, -0.3996f;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
}

bool conv1d_pytorch_match()
{
    int inChannels = 3;
    int outChannels = 1;
    int kernelSize = 1;

    std::ifstream f("../test_data/conv1d.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    Conv1d obj( inChannels, outChannels, kernelSize, true);
    obj.loadStateDict( state_dict );

    RowMatrixXf x(3, 2);
    x << 0.f, 1.f, 2.f, 3.f, 4.f, 5.f;

    auto pred = obj.forward(x);

    std::cout << "Conv1d Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(1,2);
    target << 0.6171f, 1.1664f;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
}

bool causaldilatedconv1d_pytorch_match()
{
    int inChannels = 1;
    int outChannels = 5;
    int kernelSize = 3;

    std::ifstream f("../test_data/causaldilatedconv1d.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    CausalDilatedConv1d obj( inChannels, outChannels, kernelSize, true, 2 );
    obj.loadStateDict( state_dict );

    Eigen::RowVectorXf x(4);
    x << 0.5f, -0.5f, 0.25f, -0.1f;

    auto pred = obj.forward(x);

    std::cout << "CausalDilatedConv1d Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(5,4);
    target << -0.0791f, -0.2495f, -0.3404f, -0.0638f, 
        0.4213f, 0.1339f, 0.5911f, 0.0273f, 
        -0.6683f, -0.1060f, -0.4642f, -0.3741f,
        0.1276f, -0.2533f, -0.2105f, 0.0916f,
        -0.1803f, 0.1715f, -0.1969f, 0.1731f;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
}

bool residualblock_pytorch_match()
{
    int numChannels = 5;
    int kernelSize = 3;

    std::ifstream f("../test_data/residualblock.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    ResidualBlock obj( numChannels, kernelSize, 2, true, true, true, Activation::TANH );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(5,1);
    x << 0.5f, -0.5f, 0.25f, -0.1f, 0.1f;

    auto pred = std::get<0>( obj.forward(x) );

    std::cout << "ResidualBlock Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(5,1);
    target << 0.4813f, -0.2321f, 0.2010f, 0.0285f, 0.0679f;

    return ( (pred - target).lpNorm<1>() < 1e-3 );
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
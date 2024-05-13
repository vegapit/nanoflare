#include <catch2/catch_test_macros.hpp>
#include "LSTM.h"
#include "Conv1d.h"
#include "RecNet.h"
#include "Linear.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using namespace MicroTorch;

bool lstmcell_pytorch_match()
{
    int hiddenSize = 7;
    int inputSize = 1;

    std::ifstream f("../lstmcell.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    LSTM obj( inputSize , hiddenSize, true );
    obj.loadStateDict( state_dict );

    RowMatrixXf x(2, 1);
    x << 0.5f, -0.5f;

    auto pred = obj.forward( x ); // X = [SeqLength, InputSize]
    
    std::cout << "LSTMCell Pred" << std::endl;
    std::cout << pred << std::endl;

    RowMatrixXf target(2, 7);
    target <<  -0.068939f, 0.0888935f, 0.0169113f, -0.0731596f, -0.0139663f, -0.0952078f, 0.10324f, -0.0320847f, 0.105997f, 0.0728108f, -0.0160205f, -0.060913f, -0.155882f, 0.162275f;

    return ( (pred - target).lpNorm<1>() < 1e-4 );
}

bool linear_pytorch_match()
{
    int inChannels = 3;
    int outChannels = 2;

    std::ifstream f("../linear.json");
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

    return ( (pred - target).lpNorm<1>() < 1e-4 );
}

bool conv1d_pytorch_match()
{
    int inChannels = 3;
    int outChannels = 1;
    int kernelSize = 1;

    std::ifstream f("../conv1d.json");
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

    return ( (pred - target).lpNorm<1>() < 1e-4 );
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

bool recnet_load()
{
    int hiddenSize = 32;

    std::ifstream f("../recnet.json");
    nlohmann::json data = nlohmann::json::parse(f);
    std::map<std::string, nlohmann::json> state_dict = data.get<std::map<std::string, nlohmann::json>>();

    RecNet obj(hiddenSize);
    obj.loadStateDict( state_dict );

    RowMatrixXf x(1, 4);
    x << 0.f, 1.f, 2.f, 3.f;

    auto res = obj.forward(x);

    return (res.rows() == x.rows()) && (res.cols() == x.cols());
}

TEST_CASE("convolve1d Test", "[convolve1d_calculate]")
{
    REQUIRE( convolve1d_calculate() );
}

TEST_CASE("LSTMCell Test", "[lstmcell_pytorch_match]")
{
    REQUIRE( lstmcell_pytorch_match() );
}

TEST_CASE("Conv1D Test", "[conv1d_pytorch_match]")
{
    REQUIRE( conv1d_pytorch_match() );
}

TEST_CASE("Linear Test", "[linear_pytorch_match]")
{
    REQUIRE( linear_pytorch_match() );
}

TEST_CASE("RecNet Loading Test", "[recnet_load]")
{
    REQUIRE( recnet_load() );
}
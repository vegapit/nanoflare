#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace MicroTorch;

class Timer
{
public:
    Timer() : m_timePoint(std::chrono::high_resolution_clock::now()) {}
    ~Timer() { Stop(); }

    void Stop()
    {
        auto time_point = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_timePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(time_point).time_since_epoch().count();
        std::cout << "Duration: " << (end - start) / 1000.0 << "ms" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_timePoint;
};

int main()
{
    std::ifstream fstream("../test_data/resrnn.json");
    nlohmann::json data = nlohmann::json::parse(fstream);
    auto obj = ModelBuilder::fromJson(data);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, 512);

    {
        Timer timer;
        for(int i = 0; i < 100; i++)
            obj->forward(x);
    }

    delete obj;

    return 0;
}
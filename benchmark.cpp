#include "ModelBuilder.h"
#include "models/WaveNet.h"
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
    std::shared_ptr<BaseModel> obj;
    ModelBuilder::fromJson(data, obj);

    RowMatrixXf x = Eigen::MatrixXf::Random(1, 512);

    {
        Timer timer;
        for(int i = 0; i < 100; i++)
            obj->forward(x);
    }

    std::ifstream fstream2("../test_data/wavenet.json");
    nlohmann::json data2 = nlohmann::json::parse(fstream2);
    
    WaveNet wn(1, 5, 1, 3, std::vector<int>{ 1, 2, 4, 8, 16 }, 0.f, 1.f);
    wn.loadStateDict( data2 );

    {
        Timer timer;
        for(int i = 0; i < 100; i++)
            wn.forward(x);
    }

    auto res = wn.forward(x);

    std::cout << res.rows() << " " << res.cols() << std::endl;
    
    return 0;
}
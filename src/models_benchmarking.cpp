#include <nanobench.h>
#include "ModelBuilder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <fmt/format.h>

using namespace MicroTorch;

void run_model(ankerl::nanobench::Bench* ptr, const char* name, const int num_samples) {
    std::shared_ptr<BaseModel> obj;
    RowMatrixXf x = Eigen::MatrixXf::Random(1, num_samples);

    std::ifstream fstream( fmt::format("../data/{}.json", name) );
    nlohmann::json data = nlohmann::json::parse(fstream);    
    ModelBuilder::fromJson(data, obj);
    
    ptr->run(name, [&] {
        obj->forward(x);
    });
}

void run_libtorch_model(ankerl::nanobench::Bench* ptr, const char* name, const int num_samples) {
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load( fmt::format("../data/{}.torchscript", name) );
    } catch (const c10::Error& e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, num_samples}));

    if(strcmp(name, "reslstm") == 0) {
        std::tuple<torch::jit::IValue, torch::jit::IValue> hc { torch::zeros({1, 1, 64}) , torch::zeros({1, 1, 64}) };
        inputs.push_back( hc );
    } else if (strcmp(name, "resgru") == 0) {
        inputs.push_back( torch::zeros({1, 1, 64}) );
    }

    torch::NoGradGuard no_grad;
    module.eval();
    
    ptr->run( fmt::format("{}_libtorch", name), [&] {
        module.forward(inputs);
    });
}

int main() {
    constexpr int num_samples = 256;

    std::vector<const char*> model_names = {
        "convwaveshaper", "resgru", "reslstm", "microtcn", "tcn", "wavenet"
    };

    auto bench = ankerl::nanobench::Bench();
    bench.timeUnit( std::chrono::milliseconds(1), "ms" );
    bench.minEpochIterations(10);
    bench.epochs(100);

    for(const auto* name: model_names) {
        run_model(&bench, name, num_samples);
        run_libtorch_model(&bench, name, num_samples);
    }

    return 0;
}
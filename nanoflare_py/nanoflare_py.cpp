#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <fstream>
#include <memory>
#include <filesystem>
#include "nanoflare/ModelBuilder.h"
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/models/MicroTCN.h"
#include "nanoflare/models/ResRNN.h"
#include "nanoflare/models/HammersteinWiener.h"
#include "nanoflare/models/TCN.h"
#include "nanoflare/models/WaveNet.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/GRU.h"

namespace py = pybind11;

void register_models()
{
    using namespace Nanoflare;
    registerModel<HammersteinWiener>("HammersteinWiener");
    registerModel<MicroTCN>("MicroTCN");
    registerModel<ResRNN<GRU>>("ResGRU");
    registerModel<ResRNN<LSTM>>("ResLSTM");
    registerModel<TCN>("TCN");
    registerModel<WaveNet>("WaveNet");
}

class PyNanoflareModel
{
public:
    PyNanoflareModel(const std::string& json_path)
    {
        std::ifstream model_file( json_path );
        if (!model_file.is_open())
            throw std::runtime_error("Could not open model file: " + json_path);
        nlohmann::json j = nlohmann::json::parse( model_file );
        Nanoflare::ModelBuilder::getInstance().buildModel(j, m_model);
        if (!m_model)
            throw std::runtime_error("Failed to build model from JSON");
    }

    py::array_t<float> infer(py::array_t<float> input) 
    {
        auto buf = input.request();
        Eigen::Map<const Nanoflare::RowMatrixXf> in_mat(
            static_cast<float*>(buf.ptr),
            buf.shape[0],
            buf.shape[1]
        );

        Nanoflare::RowMatrixXf out_mat = Nanoflare::RowMatrixXf::Zero(m_model->getOutChannels(), in_mat.cols());
        m_model->forward(in_mat, out_mat);

        // Allocate a new py::array and copy
        py::array_t<float> result({out_mat.rows(), out_mat.cols()});

        // Map the result buffer and copy using Eigen
        Eigen::Map<Nanoflare::RowMatrixXf> result_map(
            result.mutable_data(),
            out_mat.rows(),
            out_mat.cols()
        );
        result_map = out_mat;

        return result;
    }

    private:
        std::shared_ptr<Nanoflare::BaseModel> m_model;
};

PYBIND11_MODULE(nanoflare_py, m)
{
    m.doc() = "Nanoflare C++ library bindings";

    py::class_<PyNanoflareModel>(m, "NanoflareModel")
        .def(py::init<const std::string&>(), py::arg("json_path"))
        .def("infer", &PyNanoflareModel::infer, py::arg("input"));

    m.def("register_models", &register_models, "Register available Nanoflare models");
}

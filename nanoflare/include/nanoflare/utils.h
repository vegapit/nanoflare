#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace NanoFlare
{

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

    inline std::vector<RowMatrixXf> loadTensor( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();

        std::vector<RowMatrixXf> tensor;

        // Loop through the vector and assign elements to rows
        for (auto i = 0; i < shape[0]; ++i)
        {
            // Create the Eigen matrix
            RowMatrixXf matrix(shape[1], shape[2]);
            for (auto j = 0; j < shape[1]; ++j)
                for (auto k = 0; k < shape[2]; ++k)
                    matrix(j, k) = values[i * shape[1] * shape[2] + j * shape[2] + k];
            tensor.push_back( matrix );
        }

        return tensor;
    }

    inline RowMatrixXf loadMatrix( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<RowMatrixXf>( values.data(), shape[0], shape[1] );
    }

    inline Eigen::VectorXf loadVector( std::string name, std::map<std::string, nlohmann::json> state_dict )
    {
        auto data = state_dict.at(name).get<std::map<std::string, nlohmann::json>>();
        auto shape = data.at("shape").get<std::vector<size_t>>();
        auto values = data.at("values").get<std::vector<float>>();
        return Eigen::Map<Eigen::VectorXf>( values.data(), shape[0] );
    } 

    inline Eigen::RowVectorXf dilate(const Eigen::Ref<const Eigen::RowVectorXf>& in, size_t dilation) noexcept
    {
        size_t in_size = in.size();
        size_t size = dilation * (in_size - 1) + 1;
        Eigen::RowVectorXf out = Eigen::RowVectorXf::Zero(size);

        const float* in_ptr = in.data();
        float* out_ptr = out.data();

        for(auto i = 0; i < in_size - 1; i++)
            out_ptr[dilation * i] = in_ptr[i];
        out_ptr[size - 1] = in_ptr[in_size - 1];
        return out;
    }

    inline Eigen::RowVectorXf convolve1d(const Eigen::Ref<const Eigen::RowVectorXf>& in, const Eigen::Ref<const Eigen::RowVectorXf>& weights) noexcept
    {
        size_t weights_size = weights.size();
        size_t out_size = in.size() - weights_size + 1;
        Eigen::RowVectorXf out = Eigen::RowVectorXf::Zero(out_size);

        for (size_t i = 0; i < out_size; ++i)
            out(i) = in.segment(i, weights_size).cwiseProduct(weights).sum();
        return out;
    }

    inline Eigen::RowVectorXf dilatedcausalconvolve1d(const Eigen::Ref<const Eigen::RowVectorXf>& in, const Eigen::Ref<const Eigen::RowVectorXf>& weights, size_t dilation) noexcept
    {   
        size_t weights_size = weights.size();
        size_t left_padding = dilation * ( weights_size - 1 );
        size_t out_size = in.size();
        Eigen::RowVectorXf out = Eigen::RowVectorXf::Zero(out_size);

        const float* in_ptr = in.data();
        const float* weights_ptr = weights.data();
        float* out_ptr = out.data();

        for(auto i = 0; i < out_size; i++)
            for(auto k = 0; k < weights_size; k++)
                if(i + k * dilation >= left_padding) // Avoid adding zero inputs
                    out_ptr[i] += weights_ptr[k] * in_ptr[i - left_padding + k * dilation];
        return out;
    }

    // Model configuration

    enum ModelType {
        MICRO_TCNET,
        RES_LSTM,
        RES_GRU,
        CONVWAVESHAPER,
        WAVENET,
        TCNET
    };

    NLOHMANN_JSON_SERIALIZE_ENUM( ModelType, {
        {MICRO_TCNET, "MicroTCN"},
        {RES_LSTM, "ResLSTM"},
        {RES_GRU, "ResGRU"},
        {CONVWAVESHAPER, "ConvWaveshaper"},
        {TCNET, "TCN"},
        {WAVENET, "WaveNet"}
    })

    struct RNNParameters
    {
        size_t input_size, hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers;
    };

    inline void from_json(const nlohmann::json& j, RNNParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    struct WaveNetParameters
    {
        size_t input_size, num_channels, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers;
        bool gated;
        std::vector<size_t> dilations;
    };

    inline void from_json(const nlohmann::json& j, WaveNetParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("num_channels").get_to(obj.num_channels);
        j.at("output_size").get_to(obj.output_size);
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("dilations").get_to(obj.dilations);
        j.at("stack_size").get_to(obj.stack_size);
        j.at("gated").get_to(obj.gated);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    struct TCNParameters
    {
        size_t input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers;
    };

    inline void from_json(const nlohmann::json& j, TCNParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("stack_size").get_to(obj.stack_size);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    struct MicroTCNParameters
    {
        size_t input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers;
    };

    inline void from_json(const nlohmann::json& j, MicroTCNParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("stack_size").get_to(obj.stack_size);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    struct ConvWaveshaperParameters
    {
        size_t kernel_size, depth_size, num_channels;
    };

    inline void from_json(const nlohmann::json& j, ConvWaveshaperParameters& obj) {
        j.at("kernel_size").get_to(obj.kernel_size);
        j.at("depth_size").get_to(obj.depth_size);
        j.at("num_channels").get_to(obj.num_channels);
    }

    struct ModelConfig
    {
        ModelType model_type;
        float norm_mean, norm_std;
    };

    inline void from_json(const nlohmann::json& j, ModelConfig& obj) {
        obj.model_type = j.at("model_type").template get<ModelType>();
        j.at("norm_mean").get_to(obj.norm_mean);
        j.at("norm_std").get_to(obj.norm_std);
    }

}
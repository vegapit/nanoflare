#pragma once

#include <nlohmann/json.hpp>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/models/MicroTCN.h"
#include "nanoflare/models/ResRNN.h"
#include "nanoflare/models/ConvWaveshaper.h"
#include "nanoflare/models/TCN.h"
#include "nanoflare/models/WaveNet.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/layers/GRU.h"

namespace Nanoflare
{

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

    struct ModelBuilder
    {
        static void fromJson(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();

            switch (config.model_type)
            {
                case CONVWAVESHAPER: {
                    auto parameters = data.at("parameters").template get<ConvWaveshaperParameters>();
                    model = std::make_shared<ConvWaveshaper>(parameters.kernel_size, parameters.depth_size, parameters.num_channels, config.norm_mean, config.norm_std);
                    break;
                }
                case MICRO_TCNET: {
                    auto parameters = data.at("parameters").template get<MicroTCNParameters>();
                    model = std::make_shared<MicroTCN>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.kernel_size, parameters.stack_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
                    break;
                }
                case RES_LSTM: {
                    auto parameters = data.at("parameters").template get<ResRNNParameters>();
                    model = std::make_shared<ResRNN<LSTM>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
                    break;
                }
                case RES_GRU: {
                    auto parameters = data.at("parameters").template get<ResRNNParameters>();
                    model = std::make_shared<ResRNN<GRU>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
                    break;
                }
                case TCNET: {
                    auto parameters = data.at("parameters").template get<TCNParameters>();
                    model = std::make_shared<TCN>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.kernel_size, parameters.stack_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
                    break;
                }
                case WAVENET: {
                    auto parameters = data.at("parameters").template get<WaveNetParameters>();
                    model = std::make_shared<WaveNet>(parameters.input_size, parameters.num_channels, parameters.output_size, parameters.kernel_size, parameters.dilations, parameters.stack_size, parameters.gated, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
                    break;
                }
                default:
                    return;
            }

            model->loadStateDict( state_dict ); 
        }
    };

}
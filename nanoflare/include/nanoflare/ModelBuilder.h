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

    struct ModelConfig
    {
        float norm_mean, norm_std;
        std::map<std::string, nlohmann::json> meta_data;
    };

    inline void from_json(const nlohmann::json& j, ModelConfig& obj)
    {
        obj.meta_data = j.at("meta_data").template get<std::map<std::string, nlohmann::json>>();
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

            std::string model_type;
            config.meta_data["model_type"].get_to(model_type);

            if( model_type == "ConvWaveshaper")
            {
                auto parameters = data.at("parameters").template get<ConvWaveshaperParameters>();
                model = std::make_shared<ConvWaveshaper>(parameters.kernel_size, parameters.depth_size, parameters.num_channels, config.norm_mean, config.norm_std);
            }
            else if( model_type == "MicroTCN" )
            {
                auto parameters = data.at("parameters").template get<MicroTCNParameters>();
                model = std::make_shared<MicroTCN>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.kernel_size, parameters.stack_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            }
            else if( model_type == "ResGRU" )
            {
                auto parameters = data.at("parameters").template get<ResRNNParameters>();
                model = std::make_shared<ResRNN<GRU>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            }
            else if( model_type == "ResLSTM" )
            {
                auto parameters = data.at("parameters").template get<ResRNNParameters>();
                model = std::make_shared<ResRNN<LSTM>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            }
            else if( model_type == "TCN" )
            {
                auto parameters = data.at("parameters").template get<TCNParameters>();
                model = std::make_shared<TCN>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.kernel_size, parameters.stack_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            }
            else if( model_type == "WaveNet" )
            {
                auto parameters = data.at("parameters").template get<WaveNetParameters>();
                model = std::make_shared<WaveNet>(parameters.input_size, parameters.num_channels, parameters.output_size, parameters.kernel_size, parameters.dilations, parameters.stack_size, parameters.gated, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            }
            else { return; }

            model->loadStateDict( state_dict ); 
        }
    };

}
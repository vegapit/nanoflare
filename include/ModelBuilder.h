#pragma once

#include <nlohmann/json.hpp>
#include "BaseModel.h"
#include "models/ResRNN.h"
#include "models/WaveNet.h"
#include "LSTM.h"
#include "GRU.h"
#include "utils.h"

namespace MicroTorch
{

    struct ModelBuilder
    {
        static void fromJson(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();

            switch (config.model_type)
            {
                case RES_LSTM: {
                    auto parameters = data.at("parameters").template get<RNNParameters>();
                    model = std::make_shared<ResRNN<LSTM>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.rnn_bias, parameters.linear_bias, config.norm_mean, config.norm_std);
                    break;
                }
                case RES_GRU: {
                    auto parameters = data.at("parameters").template get<RNNParameters>();
                    model = std::make_shared<ResRNN<GRU>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.rnn_bias, parameters.linear_bias, config.norm_mean, config.norm_std);
                    break;
                }
                case WAVENET: {
                    auto parameters = data.at("parameters").template get<WaveNetParameters>();
                    model = std::make_shared<WaveNet>(parameters.input_size, parameters.num_channels, parameters.output_size, parameters.kernel_size, parameters.dilations, parameters.stack_size, config.norm_mean, config.norm_std);
                    break;
                }
                default:
                    return;
            }

            model->loadStateDict( state_dict ); 
        }
    };

}
#pragma once

#include <nlohmann/json.hpp>
#include "BaseModel.h"
#include "models/ResRNN.h"
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

            auto model_def = data.at("model_def").template get<ModelDef>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();

            switch (model_def.type)
            {
                case RES_LSTM:
                    model = std::make_shared<ResRNN<LSTM>>(model_def.input_size, model_def.hidden_size, model_def.output_size, model_def.rnn_bias, model_def.linear_bias, model_def.norm_mean, model_def.norm_std);
                    break;
                case RES_GRU:
                    model = std::make_shared<ResRNN<GRU>>(model_def.input_size, model_def.hidden_size, model_def.output_size, model_def.rnn_bias, model_def.linear_bias, model_def.norm_mean, model_def.norm_std);
                    break;
                default:
                    return;
            }

            model->loadStateDict( state_dict ); 
        }
    };

}
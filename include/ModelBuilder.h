#pragma once

#include <fstream>
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
        static BaseModel* fromJson(std::string file_path)
        {
            std::ifstream fstream(file_path);
            nlohmann::json data = nlohmann::json::parse(fstream);
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto model_def = data.at("model_def").template get<ModelDef>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();

            std::cout << model_def.input_size << std::endl;

            switch (model_def.type)
            {
                case RES_LSTM:
                    return new ResRNN<LSTM>(model_def.input_size, model_def.hidden_size, model_def.output_size, model_def.rnn_bias, model_def.linear_bias);
                case RES_GRU:
                    return new ResRNN<GRU>(model_def.input_size, model_def.hidden_size, model_def.output_size, model_def.rnn_bias, model_def.linear_bias);
                default:
                    return nullptr;
            }
        }
    };

}
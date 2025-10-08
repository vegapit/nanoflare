#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/Linear.h"
#include "nanoflare/layers/LSTM.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    struct HammersteinWienerParameters
    {
        size_t input_size, linear_input_size, linear_output_size, hidden_size, output_size;
    };

    inline void from_json(const nlohmann::json& j, HammersteinWienerParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("linear_input_size").get_to(obj.linear_input_size);
        j.at("linear_output_size").get_to(obj.linear_output_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
    }

    class HammersteinWiener : public BaseModel
    {
    public:
        HammersteinWiener(size_t input_size, size_t linear_input_size, size_t linear_output_size, size_t hidden_size, size_t output_size, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_inputLinear(input_size, linear_input_size, true),
            m_lstm(linear_input_size, linear_output_size, true),
            m_hiddenLinear(linear_output_size, hidden_size, true), 
            m_outputLinear(hidden_size, output_size, true),
            m_skipLinear(hidden_size, output_size, false)
        {}
        ~HammersteinWiener() = default;

        inline RowMatrixXf forward( const Eigen::Ref<const RowMatrixXf>& x ) noexcept override final
        {
            m_norm_x = x;
            normalise( m_norm_x );
            m_norm_x = m_inputLinear.forward( m_norm_x.transpose() ).array().tanh();
            m_norm_x = m_lstm.forward( m_norm_x );
            m_norm_x = m_hiddenLinear.forward( m_norm_x ).array().tanh();
            return m_skipLinear.forwardTranspose( x ) + m_outputLinear.forward( m_norm_x ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            auto input_linear_state_dict = state_dict[std::string("input_linear")].get<std::map<std::string, nlohmann::json>>();
            m_inputLinear.loadStateDict( input_linear_state_dict );
            auto lstm_state_dict = state_dict[std::string("lstm")].get<std::map<std::string, nlohmann::json>>();
            m_lstm.loadStateDict( lstm_state_dict );
            auto hidden_linear_state_dict = state_dict[std::string("hidden_linear")].get<std::map<std::string, nlohmann::json>>();
            m_hiddenLinear.loadStateDict( hidden_linear_state_dict );
            auto output_linear_state_dict = state_dict[std::string("output_linear")].get<std::map<std::string, nlohmann::json>>();
            m_outputLinear.loadStateDict( output_linear_state_dict );
            auto skip_linear_state_dict = state_dict[std::string("skip_linear")].get<std::map<std::string, nlohmann::json>>();
            m_skipLinear.loadStateDict( skip_linear_state_dict );
        }

        void resetState() { m_lstm.resetState(); }

        static void build(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();
            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();
            auto parameters = data.at("parameters").template get<HammersteinWienerParameters>();
            model = std::make_shared<HammersteinWiener>(parameters.input_size, parameters.linear_input_size, parameters.linear_output_size, parameters.hidden_size, parameters.output_size, config.norm_mean, config.norm_std);
            model->loadStateDict( state_dict );
        }

    private:
        Linear m_inputLinear, m_hiddenLinear, m_outputLinear, m_skipLinear;
        LSTM m_lstm;
        mutable RowMatrixXf m_norm_x;
    };

}
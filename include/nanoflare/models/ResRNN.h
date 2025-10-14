#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include "nanoflare/models/BaseModel.h"
#include "nanoflare/layers/PlainSequential.h"
#include "nanoflare/utils.h"

namespace Nanoflare
{
    struct ResRNNParameters
    {
        size_t input_size, hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers;
    };

    inline void from_json(const nlohmann::json& j, ResRNNParameters& obj) {
        j.at("input_size").get_to(obj.input_size);
        j.at("hidden_size").get_to(obj.hidden_size);
        j.at("output_size").get_to(obj.output_size);
        j.at("ps_hidden_size").get_to(obj.ps_hidden_size);
        j.at("ps_num_hidden_layers").get_to(obj.ps_num_hidden_layers);
    }

    template<class T> // T can be of type LSTM or GRU
    class ResRNN : public BaseModel
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ResRNN(size_t input_size, size_t hidden_size, size_t output_size,  size_t ps_hidden_size, size_t ps_num_hidden_layers, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_rnn(input_size, hidden_size, true), 
            m_plainSequential( hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers)
        {}
        ~ResRNN() = default;

        inline RowMatrixXf forward( const Eigen::Ref<const RowMatrixXf>& x ) noexcept override final
        {
            m_norm_x = x;
            normalise( m_norm_x );
            RowMatrixXf y = RowMatrixXf::Zero( x.rows(), x.cols() );
            m_rnn.forward( m_norm_x.transpose(), y );
            m_plainSequential.forward( y, y );
            return x + y.transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override final
        {
            auto lstm_state_dict = state_dict[std::string("rnn")].get<std::map<std::string, nlohmann::json>>();
            m_rnn.loadStateDict( lstm_state_dict );
            auto ps_state_dict = state_dict[std::string("plain_sequential")].get<std::map<std::string, nlohmann::json>>();
            m_plainSequential.loadStateDict( ps_state_dict );
        }

        void resetState() { m_rnn.resetState(); }

        static void build(const nlohmann::json& data, std::shared_ptr<BaseModel>& model)
        {
            auto doc = data.get<std::map<std::string, nlohmann::json>>();

            auto config = data.at("config").template get<ModelConfig>();
            auto state_dict = data.at("state_dict").get<std::map<std::string, nlohmann::json>>();
            auto parameters = data.at("parameters").template get<ResRNNParameters>();
            model = std::make_shared<ResRNN<T>>(parameters.input_size, parameters.hidden_size, parameters.output_size, parameters.ps_hidden_size, parameters.ps_num_hidden_layers, config.norm_mean, config.norm_std);
            model->loadStateDict( state_dict );
        }

    private:
        T m_rnn;
        PlainSequential m_plainSequential;
        mutable RowMatrixXf m_norm_x;
    };

}
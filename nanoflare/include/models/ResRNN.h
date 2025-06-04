#pragma once

#include <Eigen/Dense>
#include "models/BaseModel.h"
#include "PlainSequential.h"
#include "utils.h"

namespace NanoFlare
{

    template<class T> // T can be of type LSTM or GRU
    class ResRNN : public BaseModel
    {
    public:
        ResRNN(size_t input_size, size_t hidden_size, size_t output_size,  size_t ps_hidden_size, size_t ps_num_hidden_layers, float norm_mean, float norm_std) : BaseModel(norm_mean, norm_std), 
            m_rnn(input_size, hidden_size, true), 
            m_plainSequential( hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers)
        {}
        ~ResRNN() = default;

        inline RowMatrixXf forward( const Eigen::Ref<RowMatrixXf>& x ) noexcept override final
        {
            RowMatrixXf norm_x( x );
            normalise( norm_x );
            RowMatrixXf t_norm = norm_x.transpose();
            RowMatrixXf y = m_rnn.forward( t_norm );
            return norm_x + m_plainSequential.forward( y ).transpose();
        }

        void loadStateDict(std::map<std::string, nlohmann::json> state_dict) override
        {
            auto lstm_state_dict = state_dict[std::string("rnn")].get<std::map<std::string, nlohmann::json>>();
            m_rnn.loadStateDict( lstm_state_dict );
            auto ps_state_dict = state_dict[std::string("plain_sequential")].get<std::map<std::string, nlohmann::json>>();
            m_plainSequential.loadStateDict( ps_state_dict );
        }

        void resetState() { m_rnn.resetState(); }

    private:
        T m_rnn;
        PlainSequential m_plainSequential;
    };

}